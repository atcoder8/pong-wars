use nannou::prelude::*;
use ndarray::prelude::*;
use rand::{RngExt, rngs::ThreadRng};

/// ウィンドウの幅
const WINDOW_WIDTH: u32 = 800;
/// ウィンドウの高さ
const WINDOW_HEIGHT: u32 = 800;

/// ウィンドウ上の盤面の位置
const BOARD_POSITION: Vec2 = vec2(0.0, 0.0);
/// 盤面のサイズ
const BOARD_SIZE: f32 = 800.0;
/// 盤面上の座標の最小値
const BOARD_MIN_POINT: Vec2 = vec2(-BOARD_SIZE / 2.0, -BOARD_SIZE / 2.0);
/// 盤面上の座標の最大値
const BOARD_MAX_POINT: Vec2 = vec2(
    BOARD_MIN_POINT.x + BOARD_SIZE,
    BOARD_MIN_POINT.y + BOARD_SIZE,
);

/// 盤面一辺あたりのタイル数
const TILES_PER_SIDE: usize = 16;
/// タイルの一辺の長さ
const TILE_SIZE: f32 = BOARD_SIZE / TILES_PER_SIDE as f32;

/// チーム数
const NUM_TEAMS: usize = 4;
/// チーム内のボールの個数
const BALLS_PER_TEAM: usize = 3;
/// ボールの個数の合計
const TOTAL_NUM_BALLS: usize = NUM_TEAMS * BALLS_PER_TEAM;

/// ボールの半径
const BALL_RADIUS: f32 = 15.0;
/// ボールの速度
const BALL_SPEED: f32 = 800.0;

fn main() {
    nannou::app(model).update(update).run();
}

#[derive(Debug, Clone, Copy)]
struct Ball {
    team_id: usize,
    radius: f32,
    point: Vec2,
    velocity: Vec2,
}

#[derive(Debug, Clone)]
struct Model {
    is_running: bool,
    balls: [Ball; TOTAL_NUM_BALLS],
    board: Array2<usize>,
}

fn model(app: &App) -> Model {
    let mut rng = rand::rng();

    app.new_window()
        .title("Pong Wars")
        .size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .view(view)
        .key_pressed(key_pressed)
        .build();

    let sqrt_floor_num_teams = NUM_TEAMS.isqrt();
    // N^2≥NUM_TEAMS となるように盤面上のタイルを N×N ブロックに分割
    let blocks_by_side = sqrt_floor_num_teams + (sqrt_floor_num_teams.pow(2) < NUM_TEAMS) as usize;
    // ブロック内の一辺ごとのタイル数
    let tiles_per_side_in_block = TILES_PER_SIDE.div_ceil(blocks_by_side);

    let decide_ball_point = |rng: &mut ThreadRng, team_id: usize| {
        let block_pos = [team_id % blocks_by_side, team_id / blocks_by_side];

        let block_min_point = BOARD_MIN_POINT
            + Vec2::from_array(block_pos.map(|t| (tiles_per_side_in_block * t) as f32 * TILE_SIZE));
        let block_max_point = BOARD_MIN_POINT
            + Vec2::from_array(block_pos.map(|t| {
                (tiles_per_side_in_block * (t + 1)).min(TILES_PER_SIDE) as f32 * TILE_SIZE
            }));
        let placeable_rect = Rect::from_corners(block_min_point, block_max_point).pad(BALL_RADIUS);
        let lower_pos = placeable_rect.bottom_left();
        let upper_pos = placeable_rect.top_right();
        Vec2 {
            x: rng.random_range(lower_pos.x..upper_pos.x),
            y: rng.random_range(lower_pos.y..upper_pos.y),
        }
    };

    let balls: [Ball; TOTAL_NUM_BALLS] = std::array::from_fn(|i| {
        let team_id = i / BALLS_PER_TEAM;
        Ball {
            team_id,
            radius: BALL_RADIUS,
            point: decide_ball_point(&mut rng, team_id),
            velocity: BALL_SPEED * Vec2::from_angle(rng.random_range(0.0..f32::TAU())),
        }
    });

    let board = Array2::from_shape_fn([TILES_PER_SIDE; 2], |(row, col)| {
        let block_row = row / tiles_per_side_in_block;
        let block_col = col / tiles_per_side_in_block;
        (blocks_by_side * block_row + block_col).min(NUM_TEAMS - 1)
    });

    Model {
        is_running: false,
        balls,
        board,
    }
}

/// ボールとタイルの衝突を検出します。
fn detect_collisions_with_tiles(
    ball: Ball,
    board: &Array2<usize>,
) -> Option<((usize, usize), Vec2)> {
    // ボールと衝突している可能性のあるタイルの範囲を計算
    let lower_point = ball.point - ball.radius;
    let upper_point = ball.point + ball.radius;
    let [lower_tile_col, lower_tile_row] = ((lower_point - BOARD_MIN_POINT) / TILE_SIZE)
        .to_array()
        .map(|num_front_tiles| {
            (num_front_tiles as isize).clamp(0, TILES_PER_SIDE as isize - 1) as usize
        });
    let [upper_tile_col, upper_tile_row] = ((upper_point - BOARD_MIN_POINT) / TILE_SIZE)
        .to_array()
        .map(|num_front_tiles| {
            (num_front_tiles as isize).clamp(0, TILES_PER_SIDE as isize - 1) as usize
        });

    board
        .slice(s![
            lower_tile_row..=upper_tile_row,
            lower_tile_col..=upper_tile_col
        ])
        .indexed_iter()
        .find_map(|((row, col), &tile_owner)| {
            let row = row + lower_tile_row;
            let col = col + lower_tile_col;

            // 自陣のタイルには衝突しない
            if tile_owner == ball.team_id {
                return None;
            }

            let tile_min_point =
                BOARD_MIN_POINT + Vec2::from_array([col, row].map(|t| t as f32 * TILE_SIZE));
            let tile_max_point = tile_min_point + TILE_SIZE;
            // ボールの中心から最も近いタイル内部の点 (衝突している場合は衝突点となる)
            let nearest_point = ball.point.clamp(tile_min_point, tile_max_point);

            let (unit_normal, distance) = (nearest_point - ball.point).normalize_and_length();

            // ボールの中心が衝突したタイルの内部に存在する場合は速度を変えない
            if distance == 0.0 {
                return Some(((row, col), ball.velocity));
            }

            let dot_product = ball.velocity.dot(unit_normal);

            // 距離がボールの半径より大きい場合や離れる方向に向かっている場合は衝突しない
            if distance > ball.radius || dot_product < 0.0 {
                return None;
            }

            let bounced_velocity = ball.velocity - 2.0 * dot_product * unit_normal;
            Some(((row, col), bounced_velocity))
        })
}

/// ボールと外枠の衝突を検出します。
fn detect_collisions_with_outer_frame(ball: Ball) -> Option<Vec2> {
    if (ball.point.x - ball.radius < BOARD_MIN_POINT.x && ball.velocity.x.is_negative())
        || (ball.point.x + ball.radius > BOARD_MAX_POINT.x && ball.velocity.x.is_positive())
    {
        return Some(ball.velocity.with_x(-ball.velocity.x));
    }

    if (ball.point.y - ball.radius < BOARD_MIN_POINT.y && ball.velocity.y.is_negative())
        || (ball.point.y + ball.radius > BOARD_MAX_POINT.y && ball.velocity.y.is_positive())
    {
        return Some(ball.velocity.with_y(-ball.velocity.y));
    }

    None
}

fn update(app: &App, model: &mut Model) {
    if !model.is_running {
        return;
    }

    let elapsed_seconds = app.time_delta();

    let mut updated_balls = model.balls;

    for (ball_index, ball) in model.balls.into_iter().enumerate() {
        let updated_ball = &mut updated_balls[ball_index];

        // ボールとブロックの衝突を検出
        if let Some((collided_tile_position, updated_velocity)) =
            detect_collisions_with_tiles(ball, &model.board)
        {
            model.board[collided_tile_position] = ball.team_id;
            updated_ball.velocity = updated_velocity;
        }

        // ボールと外枠の衝突を検出
        if let Some(updated_velocity) = detect_collisions_with_outer_frame(ball) {
            updated_ball.velocity = updated_velocity;
        }

        updated_ball.point += elapsed_seconds * ball.velocity;
    }

    model.balls = updated_balls;
}

fn view(app: &App, model: &Model) {
    const COLORS: [Srgba; 13] = [
        RED, BLUE, GREEN, ORANGE, FUCHSIA, OLIVE, TEAL, YELLOW, LIME, MAROON, PURPLE, AQUA, NAVY,
    ];

    let draw = app.draw();
    draw.background().color(WHITE);

    let board_draw = draw.xy(BOARD_POSITION);

    let team_to_color = |team_id: usize| COLORS[team_id % COLORS.len()];

    // タイルを描画
    for ((row, col), &tile_owner) in model.board.indexed_iter() {
        let tile_point =
            BOARD_MIN_POINT + Vec2::from_array([col, row].map(|t| (t as f32 + 0.5) * TILE_SIZE));
        let color = team_to_color(tile_owner).with_alpha(0.3);
        board_draw
            .rect()
            .xy(tile_point)
            .w_h(TILE_SIZE, TILE_SIZE)
            .color(color);
    }

    // 格子線を描画
    for row in 0..=TILES_PER_SIDE {
        let y = BOARD_MIN_POINT.y + row as f32 * TILE_SIZE;
        board_draw
            .line()
            .points(BOARD_MIN_POINT.with_y(y), BOARD_MAX_POINT.with_y(y))
            .color(BLACK);
    }
    for col in 0..=TILES_PER_SIDE {
        let x = BOARD_MIN_POINT.x + col as f32 * TILE_SIZE;
        board_draw
            .line()
            .points(BOARD_MIN_POINT.with_x(x), BOARD_MAX_POINT.with_x(x))
            .color(BLACK);
    }

    // ボールを描画
    for ball in model.balls {
        board_draw
            .ellipse()
            .xy(ball.point)
            .radius(ball.radius)
            .color(team_to_color(ball.team_id));
    }

    // 停止状態の場合はメッセージを表示
    if !model.is_running {
        board_draw
            .rect()
            .xy((BOARD_MIN_POINT + BOARD_MAX_POINT) / 2.0)
            .wh(BOARD_MAX_POINT - BOARD_MIN_POINT)
            .color(WHITE.with_alpha(0.5));
        board_draw
            .text("Press the space key to run.")
            .font_size(50)
            .color(BLACK);
    }
}

fn key_pressed(_app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::Space {
        model.is_running = !model.is_running;
    }
}
