// mcts_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>
#include <iostream>

namespace py = pybind11;

const int BOARD_SIZE = 15;
const int BOARD_AREA = BOARD_SIZE * BOARD_SIZE;

// --- Fast Gomoku Logic ---
class GomokuGame {
public:
    std::vector<int> board;
    int current_player;
    int move_count;

    GomokuGame() : board(BOARD_AREA, 0), current_player(1), move_count(0) {}

    void reset() {
        std::fill(board.begin(), board.end(), 0);
        current_player = 1;
        move_count = 0;
    }

    // 복사 생성자 (시뮬레이션용)
    GomokuGame(const GomokuGame& other) = default;

    bool step(int action) {
        if (board[action] != 0) return false;
        board[action] = current_player;
        move_count++;
        current_player *= -1;
        return true;
    }

    std::pair<bool, int> check_win(int action) {
        int r = action / BOARD_SIZE;
        int c = action % BOARD_SIZE;
        int player = board[action];
        if (player == 0) return {false, 0};

        int dr[] = {0, 1, 1, 1};
        int dc[] = {1, 0, 1, -1};

        for (int i = 0; i < 4; ++i) {
            int count = 1;
            for (int sign : {-1, 1}) {
                int nr = r + dr[i] * sign;
                int nc = c + dc[i] * sign;
                while (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && board[nr * BOARD_SIZE + nc] == player) {
                    count++;
                    nr += dr[i] * sign;
                    nc += dc[i] * sign;
                }
            }
            if (count >= 5) return {true, player};
        }
        if (move_count == BOARD_AREA) return {true, 0}; // Draw
        return {false, 0};
    }

    std::vector<int> get_legal_actions() {
        std::vector<int> legals;
        legals.reserve(BOARD_AREA - move_count);
        for (int i = 0; i < BOARD_AREA; ++i) {
            if (board[i] == 0) legals.push_back(i);
        }
        return legals;
    }

    // 신경망 입력용 3x15x15 텐서 생성
    py::array_t<float> get_state() {
        py::array_t<float> result({3, BOARD_SIZE, BOARD_SIZE});
        auto ptr = result.mutable_unchecked<3>();
        
        // 0으로 초기화
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < BOARD_SIZE; ++h)
                for (int w = 0; w < BOARD_SIZE; ++w)
                    ptr(c, h, w) = 0.0f;

        for (int i = 0; i < BOARD_AREA; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;
            if (board[i] == current_player) ptr(0, r, c) = 1.0f;
            else if (board[i] == -current_player) ptr(1, r, c) = 1.0f;
            ptr(2, r, c) = 1.0f; // Turn info (always 1 for current player perspective)
        }
        return result;
    }
};

// --- MCTS Node ---
struct Node {
    int visit_count = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    int action = -1;
    bool is_expanded = false;
    
    std::vector<std::unique_ptr<Node>> children;
    Node* parent = nullptr;

    // [수정] 초기화 리스트 순서 변경 (action이 parent보다 먼저 선언되어 있으므로 순서 맞춤)
    Node(float p, Node* parent_node, int act) : prior(p), action(act), parent(parent_node) {}

    float value() const {
        return visit_count == 0 ? 0.0f : value_sum / visit_count;
    }

    float ucb(float parent_visit_sqrt, float cpuct = 1.0f) const {
        return value() + cpuct * prior * parent_visit_sqrt / (1.0f + visit_count);
    }
};

// --- MCTS Engine ---
class MCTS {
    std::unique_ptr<Node> root;
    GomokuGame root_game;
    
    // [수정] 누락된 멤버 변수 선언 추가
    bool game_over = false;
    int winner = 0;

    // 재사용을 위한 임시 변수들
    Node* current_node;
    GomokuGame scratch_game;
    std::vector<Node*> search_path;

public:
    MCTS() {
        reset();
    }

    void reset() {
        root = std::make_unique<Node>(1.0f, nullptr, -1);
        root_game.reset();
        // [수정] 초기화
        game_over = false;
        winner = 0;
    }

    void update_root_game(int action) {
        root_game.step(action);

        auto result = root_game.check_win(action); 
        if (result.first) {
            game_over = true;
            winner = result.second;
        }
        
        // 루트를 이동시키는 로직 (기존 트리 재사용)
        Node* next_root = nullptr;
        if (root && root->is_expanded) {
            for (auto& child : root->children) {
                if (child->action == action) {
                    next_root = child.release(); // 소유권 이전
                    break;
                }
            }
        }
        
        if (next_root) {
            root.reset(next_root);
            root->parent = nullptr;
        } else {
            root = std::make_unique<Node>(1.0f, nullptr, -1);
        }
    }

    py::tuple check_game_status() {
        return py::make_tuple(game_over, winner);
    }

    // 1. Selection & Simulation (Python이 호출)
    py::object select_leaf() {
        // 이미 게임이 끝났으면 시뮬레이션 중단
        if (game_over) return py::none();

        current_node = root.get();
        scratch_game = root_game; // 복사 (비용 저렴함)
        search_path.clear();
        search_path.push_back(current_node);

        while (current_node->is_expanded) {
            float sqrt_total_visit = std::sqrt((float)current_node->visit_count);
            float best_score = -1e9;
            Node* best_child = nullptr;

            for (auto& child : current_node->children) {
                float score = child->ucb(sqrt_total_visit);
                if (score > best_score) {
                    best_score = score;
                    best_child = child.get();
                }
            }

            if (!best_child) break; // Should not happen if expanded

            current_node = best_child;
            // [수정] 리턴값 무시 (Warning 제거)
            scratch_game.step(current_node->action);
            search_path.push_back(current_node);
            
            // 이동 후 승패 체크
            auto result = scratch_game.check_win(current_node->action);
            if (result.first) {
                // 게임 끝남 (Value = 1.0 for current player)
                backpropagate_value(1.0f); 
                return py::none(); // 추론 불필요
            }
        }

        // 추론이 필요한 Leaf 상태 반환
        return scratch_game.get_state();
    }

    // 2. Expansion & Backpropagation (Python이 추론 후 호출)
    void backpropagate(py::array_t<float> policy_logits, float value) {
        if (!current_node->is_expanded) {
            auto legal_actions = scratch_game.get_legal_actions();
            auto ptr = policy_logits.unchecked<1>();
            
            float sum_prob = 0.0f;
            std::vector<float> probs;
            probs.reserve(legal_actions.size());

            for (int action : legal_actions) {
                float p = ptr(action); 
                probs.push_back(p);
                sum_prob += p;
            }

            for (size_t i = 0; i < legal_actions.size(); ++i) {
                float p = probs[i] / (sum_prob + 1e-8f);
                current_node->children.push_back(std::make_unique<Node>(p, current_node, legal_actions[i]));
            }
            current_node->is_expanded = true;
        }

        backpropagate_value(value);
    }
    
    // 내부 헬퍼
    void backpropagate_value(float value) {
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            (*it)->visit_count++;
            (*it)->value_sum += value;
            value = -value; // 플레이어 전환
        }
    }

    void add_root_noise(float alpha, float epsilon) {
        if (!root || root->children.empty()) return;
        
        std::mt19937 gen(std::random_device{}());
        std::gamma_distribution<float> d(alpha, 1.0f);
        
        std::vector<float> noise;
        float sum_noise = 0.0f;
        for (size_t i=0; i<root->children.size(); ++i) {
            float n = d(gen);
            noise.push_back(n);
            sum_noise += n;
        }
        
        for (size_t i=0; i<root->children.size(); ++i) {
            root->children[i]->prior = 
                (1 - epsilon) * root->children[i]->prior + 
                epsilon * (noise[i] / sum_noise);
        }
    }

    // 결과 반환
    py::tuple get_action_probs(float temp) {
        std::vector<float> probs(BOARD_AREA, 0.0f);
        
        if (temp == 0) { // Argmax
             int best_action = -1;
             int max_visits = -1;
             for (auto& child : root->children) {
                 if (child->visit_count > max_visits) {
                     max_visits = child->visit_count;
                     best_action = child->action;
                 }
             }
             if (best_action != -1) probs[best_action] = 1.0f;
        } else {
             float sum = 0.0f;
             for (auto& child : root->children) {
                 float p = std::pow(child->visit_count, 1.0f/temp);
                 probs[child->action] = p;
                 sum += p;
             }
             for (int i=0; i<BOARD_AREA; ++i) probs[i] /= sum;
        }
        
        return py::make_tuple(root_game.get_state(), py::array(BOARD_AREA, probs.data()));
    }
    
    int get_current_player() { return root_game.current_player; }
};

PYBIND11_MODULE(mcts_core, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<>())
        .def("reset", &MCTS::reset)
        .def("update_root_game", &MCTS::update_root_game)
        .def("select_leaf", &MCTS::select_leaf)
        .def("backpropagate", &MCTS::backpropagate)
        .def("add_root_noise", &MCTS::add_root_noise)
        .def("get_action_probs", &MCTS::get_action_probs)
        .def("check_game_status", &MCTS::check_game_status)
        .def("get_current_player", &MCTS::get_current_player);
}