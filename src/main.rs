use std::{collections::{HashSet, VecDeque}, fmt, thread::sleep, vec};

use ndarray::{Array1, Array2};
use terminal_size::{Width, Height, terminal_size};
use rand::prelude::*;

struct TerminalBuffer {
    width: u16,
    height: u16,
    buffer: Vec<char>,
}

impl TerminalBuffer {
    fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            buffer: vec![' '; (width as usize) * (height as usize)],
        }
    }

    fn clear(&mut self) {
        for ch in self.buffer.iter_mut() {
            *ch = ' ';
        }
    }

    fn set_char(&mut self, x: u16, y: u16, ch: char) {
        let idx = (y as usize) * (self.width as usize) + (x as usize);
        if idx < self.buffer.len() {
            self.buffer[idx] = ch;
        }
    }

    fn type_text(&mut self, x: u16, y: u16, text: &str) {
        for (i, ch) in text.chars().enumerate() {
            self.set_char(x + i as u16, y, ch);
        }
    }

    fn draw(&self) {
        for row in 0..self.height {
            for col in 0..self.width {
                let idx = (row as usize) * (self.width as usize) + (col as usize);
                print!("{}", self.buffer[idx]);
            }
            println!();
        }
    }

    fn draw_box(&mut self, x: u16, y: u16, w: u16, h: u16, filled: Option<char>) {
        const CHARS: [char; 6] = ['│', '─', '╭', '╮', '╰', '╯'];

        for i in 0..h {
            for j in 0..w {
                let ch = match (i, j) {
                    (i, _) if i == 0 || i == h - 1 => CHARS[1],
                    (_, j) if j == 0 || j == w - 1 => CHARS[0],
                    _ => {
                        if let Some(fill_char) = filled {
                            fill_char
                        } else {
                            continue;
                        }
                    }
                };
                self.set_char(x + j, y + i, ch);
            }
        }

        self.set_char(x, y, CHARS[2]);
        self.set_char(x + w - 1, y, CHARS[3]);
        self.set_char(x, y + h - 1, CHARS[4]);
        self.set_char(x + w - 1, y + h - 1, CHARS[5]);
    }

}

#[derive(Clone)]
struct NNet {
    layers: Vec<Layer>,
}

#[derive(Clone)]
struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

#[derive(Clone, Copy)]
struct Activation {
    name: &'static str,
    func: fn(f32) -> f32,
}

#[allow(dead_code)]
impl Activation {
    const TANH: Self = Self { name: "Tanh", func: |x| x.tanh() };
    const ARCTAN: Self = Self { name: "Arctan", func: |x| x.atan() };
    const EXP: Self = Self { name: "Exp", func: |x| 2.0 / (1.0 + (-x).exp()) - 1.0 };
    const APROX: Self = Self { 
        name: "Aprox", 
        func: |x| {
            if x < -2.5 { -1.0 }
            else if x > 2.5 { 1.0 }
            else { 0.4 * x }
        },
    };
    const NONE: Self = Self { name: "None", func: |x| x };
    const SIN: Self = Self { name: "Sin", func: |x| x.sin() };
    const HARD_STEP: Self = Self { name: "HardStep", func: |x| if x >= 0.0 { 1.0 } else { -1.0 } };

    fn apply(&self, x: f32) -> f32 {
        (self.func)(x)
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[allow(dead_code)]
impl NNet {
    fn get_reandom_weight(w: usize, h: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((h, w), |_| {
            rng.random_range(-1.0..1.0)
        })
    }

    fn get_random_biases(size: usize) -> Array1<f32> {
        let mut rng = rand::rng();
        Array1::from_shape_fn(size, |_| {
            rng.random_range(-1.0..1.0)
        })
    }

    fn new(layer_sizes: &[usize], activations: Option<&[Activation]>) -> Self {
        let mut layers = Vec::new();
        let mut rng = rand::rng();

        layers.push(Layer {
            weights: Array2::zeros((0, 0)),
            biases: Array1::zeros(0),
            activation: Activation::NONE,
        });

        for i in 1..layer_sizes.len() {
            let input_size = layer_sizes[i - 1];
            let output_size = layer_sizes[i];
            
            let weights = Array2::from_shape_fn((output_size, input_size), |_| {
                rng.random_range(-1.0..1.0)
            });
            
            let biases = Array1::from_shape_fn(output_size, |_| {
                rng.random_range(-1.0..1.0)
            });

            let activation = if let Some(acts) = activations {
                acts.get(i).cloned().unwrap_or(Activation::TANH)
            } else {
                Activation::TANH
            };

            layers.push(Layer {
                weights,
                biases,
                activation,
            });
        }

        Self { layers }
    }

    fn new_from_weights(
        layer_sizes: &[usize],
        weights: Vec<Array2<f32>>,
        biases: Vec<Array1<f32>>,
        activations: Option<&[Activation]>
    ) -> Self {
        let mut layers = Vec::new();

        layers.push(Layer {
            weights: Array2::zeros((0, 0)),
            biases: Array1::zeros(0),
            activation: Activation::NONE,
        });

        for i in 1..layer_sizes.len() {
            let layer_weights = if i - 1 < weights.len() {
                weights[i - 1].clone()
            } else {
                Self::get_reandom_weight(layer_sizes[i-1], layer_sizes[i])
            };

            let layer_biases = if i - 1 < biases.len() {
                biases[i - 1].clone()
            } else {
                Self::get_random_biases(layer_sizes[i])
            };

            let activation = if let Some(acts) = activations {
                acts.get(i).cloned().unwrap_or(Activation::TANH)
            } else {
                Activation::TANH
            };

            layers.push(Layer {
                weights: layer_weights,
                biases: layer_biases,
                activation,
            });
        }

        Self { layers }
    }

    fn get_dims(&self) -> Vec<usize> {
        self.layers.iter().map(|layer| layer.weights.ncols()).collect()
    }

    fn get_layers_count(&self) -> usize {
        self.layers.len()
    }

    fn get_weights(&self) -> Vec<Array2<f32>> {
        let mut weights = Vec::new();
        for layer in self.layers.iter().skip(1) {
            weights.push(layer.weights.clone());
        }
        weights
    }

    fn set_weights(&mut self, weights: Vec<Array2<f32>>) {
        for (i, layer_weights) in weights.into_iter().enumerate() {
            if i < self.layers.len() {
                self.layers[i].weights = layer_weights;
            }
        }
    }

    fn get_biases(&self) -> Vec<Array1<f32>> {
        let mut biases = Vec::new();
        for layer in self.layers.iter().skip(1) {
            biases.push(layer.biases.clone());
        }
        biases
    }

    fn feedforward(&self, inputs: &[f32]) -> Vec<f32> {
        if inputs.len() != self.layers[1].weights.ncols() {
            panic!("Expected {} inputs, got {}", self.layers[0].biases.len(), inputs.len());
        }

        let mut activations = Array1::from_vec(inputs.to_vec());

        for layer in self.layers.iter().skip(1) {
            activations = &layer.weights.dot(&activations) + &layer.biases;
            
            activations.mapv_inplace(|x| layer.activation.apply(x));
        }

        activations.to_vec()
    }

    fn print_weights(&self) {
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            println!("Layer {} weights:", i);
            for (j, neuron_weights) in layer.weights.iter().enumerate() {
                println!(" Neuron {}: {:?}", j, neuron_weights);
            }
        }
    }

    fn print_activations(&self) {
        for (i, layer) in self.layers.iter().enumerate() {
            println!("Layer {} activation: {}", i, layer.activation);
        }
    }

    fn mutate_weights(
        weights_1: &mut Array2<f32>,
        weights_2: &Array2<f32>,
        biases_1: &mut Array1<f32>,
        biases_2: &Array1<f32>,
        inheritance: f32,
        mutation_rate: f32,
        mutation_amount: f32,
    ) {
        let mut rng = rand::rng();

        for (w1, w2) in weights_1.iter_mut().zip(weights_2.iter()) {
            if rng.random::<f32>() < inheritance {
                *w1 = *w2;
            }
            
            if rng.random::<f32>() < mutation_rate {
                let mutation = mutation_amount * w1.abs();
                let change: f32 = rng.random_range(-mutation..mutation);
                *w1 += change;
                *w1 = w1.clamp(-1.0, 1.0);
            }
        }

        for (b1, b2) in biases_1.iter_mut().zip(biases_2.iter()) {
            if rng.random::<f32>() < inheritance {
                *b1 = *b2;
            }
            
            if rng.random::<f32>() < mutation_rate {
                let mutation = mutation_amount * b1.abs();
                let change: f32 = rng.random_range(-mutation..mutation);
                *b1 += change;
                *b1 = b1.clamp(-1.0, 1.0);
            }
        }
    }      
}

struct Arena {
    w: u16,
    h: u16,

    x: u16,
    y: u16,
    x1: u16,
    y1: u16,
}

impl Arena {
    fn new(screen_size: (u16, u16)) -> Self {
        const ARENA_WIDTH_RATIO: f32 = 0.85;
        let width = (screen_size.0 as f32 * ARENA_WIDTH_RATIO) as u16;
        
        Self { 
            w: width,
            h: screen_size.1,
            x: (screen_size.0 - width),
            y: 0,
            x1: (screen_size.0 - width)+ width - 1,
            y1: screen_size.1,
        }
    }

    fn contains(&self, px: u16, py: u16) -> bool {
        px >= self.x && px <= self.x1 && py >= self.y && py <= self.y1
    }

    fn random_position(&self) -> (f32, f32) {
        let mut rng = rand::rng();
        let x = rng.random_range(self.x + 1..self.x1);
        let y = rng.random_range(self.y + 1..self.y1 - 1);
        (x as f32, y as f32)
    }

    fn rel_pos(&self, px: f32, py: f32) -> (f32, f32) {
        (px - self.x as f32 - self.w as f32/2., py as f32 - self.y as f32 - self.h as f32/2.)
    }
}

fn main() {
    let screen_size = {
        if let Some((Width(w), Height(h))) = terminal_size() {
            (w, h - 1)
        } else {
            (80, 24)
        }
    };

    const STEPS_PER_ITERATION: usize = 150;
    const LEARNING_EPOCHS: usize = 150;
    const PUPLATION_SIZE: usize = 100;
    const SURVIVORS: usize = 15;
    const INHERITANCE: f32 = 0.5;
    const MUTATION_RATE: f32 = 0.05;
    const MUTATION_AMOUNT: f32 = 0.1;
    const FRAME_RATE: u64 = 60;
    const ITERS_TO_LOG: usize = 1;
    const GOAL_THRESHOLD: f32 = 1.2;    
    const NETWORK_STRUCTURE: &[usize] = &[4, 10, 14, 10, 2];
    const ACTIVATIONS: &[Activation] = &[Activation::NONE, Activation::NONE, Activation::NONE, Activation::NONE, Activation::EXP];
    const GOAL_CHAR: char = '█';
    const AGENT_CHARS: &[char] = &['œ', '∑', 'ß', 'ƒ', '∆', 'æ', '@', '%', '©', 'µ'];
    
    if (SURVIVORS * SURVIVORS - 1) / 2 < PUPLATION_SIZE {
        panic!("Not enough survivors to generate the population");
    }
    let mut terminal_buffer = TerminalBuffer::new(screen_size.0, screen_size.1);
    
    let arena = Arena::new(screen_size);
    let menu_box: (u16, u16, u16, u16) = (0, 0, screen_size.0 - arena.w, screen_size.1);
    
    let params_info = vec![
        format!("Arena: (x: {}, y: {}, w: {}, h: {})", arena.x, arena.y, arena.w, arena.h),
        format!("Network Structure:{:?}", NETWORK_STRUCTURE),
        format!("Population Size:{}", PUPLATION_SIZE),
        format!("Survivors:{}", SURVIVORS),
        format!("IRate:{:.2}", INHERITANCE),
        format!("MRate:{:.2}", MUTATION_RATE),
        format!("MAmount:{:.2}", MUTATION_AMOUNT),
    ];
    
    let mut last_best_distance = -1.0;
    let mut survivors: Vec<NNet> = (0..SURVIVORS).map(|_| NNet::new(NETWORK_STRUCTURE, Some(ACTIVATIONS))).collect();

    let (mut goal_x, mut goal_y) = arena.random_position();
    let (mut goal_nx, mut goal_ny) = arena.rel_pos(goal_x, goal_y);

    let mut avg_dist_n_iters: VecDeque<f32> = VecDeque::with_capacity(ITERS_TO_LOG);
    let mut best_dist_n_iters: VecDeque<f32> = VecDeque::with_capacity(ITERS_TO_LOG);
    let mut goal_shifts = 0;
    let mut from_last_shift = 0;
    let mut iteration = 0;
    loop {

        let average_best_distance = if best_dist_n_iters.len() > 0 {
            best_dist_n_iters.iter().sum::<f32>() / best_dist_n_iters.len() as f32
        } else {
            0.0
        };

        if (average_best_distance < GOAL_THRESHOLD && best_dist_n_iters.len() == ITERS_TO_LOG) || iteration > LEARNING_EPOCHS {
            (goal_x, goal_y) = arena.random_position();
            (goal_nx, goal_ny) = arena.rel_pos(goal_x, goal_y);
            best_dist_n_iters.clear();
            goal_shifts += 1;
            from_last_shift = 0;
        }


        let mut agents_positions: Vec<(f32, f32)> = (0..PUPLATION_SIZE).map(|_| arena.random_position() as (f32, f32)).collect();
        let mut distances: Vec<(usize, f32)> = agents_positions.iter().enumerate().map(|(i, (ax, ay))| {
            let (axr, ayr) = arena.rel_pos(*ax, *ay);
            let dist = ((axr - goal_nx).powi(2) + (ayr - goal_ny).powi(2)).sqrt();
            (i, dist)
        }).collect();

        let mut agents: Vec<NNet> = {
            let mut agents: Vec<NNet> = vec![];
            let mut had_offspring: HashSet<(u32, u32)> = HashSet::new();
            let mut rng = rand::rng();

            if iteration > LEARNING_EPOCHS {
                for l in 0..PUPLATION_SIZE {
                    let idx = l % SURVIVORS;
                    agents.push(survivors[idx].clone());
                }
            }

            while agents.len() < PUPLATION_SIZE {
                let parent1_idx = rng.random_range(0..SURVIVORS) as u32;
                let parent2_idx = rng.random_range(0..SURVIVORS) as u32;

                if parent1_idx != parent2_idx && (!had_offspring.contains(&(parent1_idx, parent2_idx)) || !had_offspring.contains(&(parent2_idx, parent1_idx))) {
                    had_offspring.insert((parent1_idx, parent2_idx));
                    had_offspring.insert((parent2_idx, parent1_idx));

                    let parent1 = &survivors[parent1_idx as usize];
                    let parent2 = &survivors[parent2_idx as usize];

                    let mut weights = parent1.get_weights();
                    let weight_2 = parent2.get_weights();

                    let mut biases = parent1.get_biases();
                    let biases_2 = parent2.get_biases();

                    for i in 0..weights.len() {
                        NNet::mutate_weights(
                            &mut weights[i],
                            &weight_2[i],
                            &mut biases[i],
                            &biases_2[i],
                            INHERITANCE,
                            MUTATION_RATE,
                            MUTATION_AMOUNT,
                        );
                    }

                    let child = NNet::new_from_weights(NETWORK_STRUCTURE, weights, biases, Some(ACTIVATIONS));
                    agents.push(child);
                }
            }

            agents
        };
        
        let mut priv_frame_time: u128 = 0;
        for i in 0..STEPS_PER_ITERATION {
            let start_time = std::time::Instant::now();
            clear_terminal();
            terminal_buffer.clear();

            terminal_buffer.draw_box(arena.x, arena.y, arena.w, arena.h, None);
            terminal_buffer.draw_box(menu_box.0, menu_box.1, menu_box.2, menu_box.3, None);
            
            for (i, (ax, ay)) in agents_positions.iter().enumerate() {
                let char_idx = i % AGENT_CHARS.len();
                
                if arena.contains(*ax as u16, *ay as u16) {
                    terminal_buffer.set_char(*ax as u16, *ay as u16, AGENT_CHARS[char_idx]);
                }
                
                let info_x = 2;
                let info_y = 2 + i as u16 * 4;
                
                if info_y + 8 >= screen_size.1 {
                    continue;
                }
                
                terminal_buffer.type_text(info_x, info_y, &format!("Agent ({}) {}:", AGENT_CHARS[char_idx], i + 1));
                
                let (axr, ayr) = arena.rel_pos(*ax, *ay);
                terminal_buffer.type_text(info_x, info_y + 1, &format!(" Position: ({:.2}, {:.2})", axr, ayr));
                terminal_buffer.type_text(info_x, info_y + 2, &format!(" Distance to goal: {:.2}", distances[i].1) );
            }
            
            terminal_buffer.set_char(goal_x as u16, goal_y as u16, GOAL_CHAR);

            terminal_buffer.type_text(2, screen_size.1 - 5, &format!("Goal ({}):", GOAL_CHAR));
            terminal_buffer.type_text(2, screen_size.1 - 4, &format!(" Position: ({}, {})", goal_nx, goal_ny));
            terminal_buffer.type_text(2, screen_size.1 - 3, &format!(" Absolute: ({}, {})", goal_x, goal_y));
            

            let step_info = vec![
                format!("Iteration: {}/{}", iteration + 1, LEARNING_EPOCHS),
                format!("Step: {}/{}", i + 1, STEPS_PER_ITERATION),
                format!("Frame Time: {:.2} ms", priv_frame_time),
                format!("Avg Dist: {:.2}", avg_dist_n_iters.iter().sum::<f32>() / avg_dist_n_iters.len().max(1) as f32),
                format!("Avg Best Dist: {:.2}", average_best_distance),
                format!("Goal Shifts: {}", goal_shifts),
                format!("Iterations Since Last: {}", from_last_shift),
            ];

            let mut info_x = arena.x + 2;
            for line in step_info.iter() {
                terminal_buffer.type_text(info_x, arena.y, line);
                info_x += line.len() as u16 + 2;
            }
            
            let mut arena_info_x = arena.x + 2;
            for line in params_info.iter() {
                terminal_buffer.type_text(arena_info_x, arena.y1 - 1, line);
                arena_info_x += line.len() as u16 + 2;
            }

            let best_distance: (usize, f32) = *distances.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
            let best_agent_char = AGENT_CHARS[best_distance.0 % AGENT_CHARS.len()];

            terminal_buffer.type_text(arena_info_x, arena.y1 - 1, &format!("Best Distance ({}): {:.2}", best_agent_char, best_distance.1));

            terminal_buffer.type_text(menu_box.0 + 2, menu_box.1 + menu_box.3 - 1, &format!("Last Best Distance: {:.2}", last_best_distance));
            terminal_buffer.draw();

            let deltas: Vec<(f32, f32)> = agents.iter_mut().zip(agents_positions.iter()).map(|(agent, (ax, ay))| {
                let (axr, ayr) = arena.rel_pos(*ax, *ay);
                let output = agent.feedforward(&[axr, ayr, goal_nx, goal_ny]);
                (output[0], output[1])
            }).collect();

            deltas.iter().enumerate().for_each(|(i, (dx, dy))| {
                let (ax, ay) = agents_positions[i];
                agents_positions[i] = (ax + dx, ay + dy);
            });

            distances = agents_positions.iter().enumerate().map(|(i, (ax, ay))| {
                let (axr, ayr) = arena.rel_pos(*ax, *ay);
                let dist = ((axr - goal_nx).powi(2) + (ayr - goal_ny).powi(2)).sqrt();
                (i, dist)
            }).collect();

            priv_frame_time = start_time.elapsed().as_millis();
            let frame_duration = std::time::Duration::from_millis(1000 / FRAME_RATE);
            let time_to_sleep = frame_duration.saturating_sub(start_time.elapsed()).as_millis() as u64;

            sleep(std::time::Duration::from_millis(time_to_sleep));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        last_best_distance = distances[0].1;
        
        if avg_dist_n_iters.len() == ITERS_TO_LOG {
            avg_dist_n_iters.pop_front();
        }
        let avg = distances.iter().map(|(_, d)| *d).sum::<f32>() / distances.len() as f32;
        avg_dist_n_iters.push_back(avg);

        if best_dist_n_iters.len() == ITERS_TO_LOG {
            best_dist_n_iters.pop_front();
        }
        best_dist_n_iters.push_back(distances[0].1);

        survivors = distances.iter().take(SURVIVORS).map(|(i, _)| agents[*i].clone()).collect();

        iteration += 1;
        from_last_shift += 1;
    }
}

fn clear_terminal() {
    if cfg!(debug_assertions) {
        if cfg!(target_os = "windows") {
            std::process::Command::new("cls").status().unwrap();
        } else {
            std::process::Command::new("clear").status().unwrap();
        }
    } else {
        print!("\x1b[2J\x1b[H");
    }
}