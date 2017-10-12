
pub struct Ship {
    x: f32,
    y: f32,
}
impl Ship {
    pub fn new(x: f32, y: f32) -> Self {
        Ship { x: x, y: y }
    }
    fn move_by(&mut self, x: f32, y: f32) {
        self.x += x;
        self.y += y;
    }
    fn play(&mut self, choice: Box<[f32]>) {
        self.move_by(choice[0], choice[1]);
    }
}