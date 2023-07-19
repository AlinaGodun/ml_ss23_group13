use crate::breakout_types::*;

#[derive(Debug, serde::Serialize, Clone)]
pub struct Experiment {
    pub bricks_layout: String,
    pub ball_start_x: i32,
    pub timestamps_to_win: usize,
    pub gamma: f32,
    pub start_epsilon: f32,
    pub epsilon: f32,
    pub max_episode_num: usize,
    pub current_episode_num: usize,
    pub reward_system: String,
    pub policy_found_in_s: u64,
    pub brick_length: usize
}

impl Experiment {
    pub fn new(
        bricks_layout: &String, 
        ball_start_x: &i32, 
        timestamps_to_win: &usize,
        gamma: &f32,
        start_epsilon: &f32,
        epsilon: &f32,
        max_episode_num: &usize,
        current_episode_num: &usize,
        reward_system: &String,
        policy_found_in_s: &u64,
        brick_length: &usize) -> Self {
            Self {
                bricks_layout: bricks_layout.clone(), 
                ball_start_x: ball_start_x.clone(), 
                timestamps_to_win: timestamps_to_win.clone(),
                gamma: gamma.clone(),
                start_epsilon: start_epsilon.clone(),
                epsilon: epsilon.clone(),
                max_episode_num: max_episode_num.clone(),
                current_episode_num: current_episode_num.clone(),
                reward_system: reward_system.clone(),
                policy_found_in_s: policy_found_in_s.clone(),
                brick_length: brick_length.clone()
            }
        }
}