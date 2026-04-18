use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ultimattt::game::{Game as RustGame, Move as RustMove, Player, GameState, CellState};

#[pyclass]
#[derive(Clone)]
struct UTTTGame {
    game: RustGame,
}

#[pymethods]
impl UTTTGame {
    #[new]
    fn new() -> Self {
        UTTTGame {
            game: RustGame::new(),
        }
    }

    fn make_move(&self, global: usize, local: usize) -> PyResult<Self> {
        let m = RustMove::from_coords(global, local);
        match self.game.make_move(m) {
            Ok(new_game) => Ok(UTTTGame { game: new_game }),
            Err(e) => Err(PyValueError::new_err(format!("Invalid move: {:?}", e))),
        }
    }

    fn legal_moves(&self) -> Vec<(usize, usize)> {
        self.game
            .all_moves()
            .map(|m| (m.global(), m.local()))
            .collect()
    }

    fn is_terminal(&self) -> bool {
        self.game.game_over()
    }

    fn get_result(&self) -> Option<i8> {
        match self.game.game_state() {
            GameState::Won(Player::X) => Some(1),
            GameState::Won(Player::O) => Some(-1),
            GameState::Drawn => Some(0),
            GameState::InPlay => None,
        }
    }

    fn current_player(&self) -> i8 {
        match self.game.player() {
            Player::X => 1,
            Player::O => -1,
        }
    }

    fn get_state(&self) -> Vec<Vec<Vec<i8>>> {
        // Returns a 9x9x3 representation:
        // Channel 0: X pieces
        // Channel 1: O pieces
        // Channel 2: Board state (valid moves)
        let mut state = vec![vec![vec![0i8; 9]; 9]; 3];

        for board in 0..9 {
            for cell in 0..9 {
                match self.game.at(board, cell) {
                    CellState::Played(Player::X) => state[0][board][cell] = 1,
                    CellState::Played(Player::O) => state[1][board][cell] = 1,
                    CellState::Empty => {}
                }
            }
        }

        // Mark legal moves in channel 2
        for (board, cell) in self.legal_moves() {
            state[2][board][cell] = 1;
        }

        state
    }

    fn clone_game(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.game)
    }
}

#[pymodule]
fn uttt_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UTTTGame>()?;
    Ok(())
}
