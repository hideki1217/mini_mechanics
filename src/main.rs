use std::{
    f32::consts::PI,
    ops::{Add, Div, Mul, Neg, Sub},
    thread::sleep,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy)]
struct RGB(u8, u8, u8);
impl RGB {
    fn black() -> RGB {
        RGB(0, 0, 0)
    }
    fn white() -> RGB {
        RGB(255, 255, 255)
    }
    fn red() -> RGB {
        RGB(255, 0, 0)
    }
    fn green() -> RGB {
        RGB(0, 255, 0)
    }
    fn blue() -> RGB {
        RGB(0, 0, 255)
    }
}

#[derive(Debug, Clone, Copy)]
struct Vec3(f32, f32, f32);
impl Vec3 {
    pub fn norm(&self) -> f32 {
        (self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt()
    }

    pub fn sum(&self) -> f32 {
        self.0 + self.1 + self.2
    }

    pub fn inner(&self, rhs: &Vec3) -> f32 {
        (self * rhs).sum()
    }

    pub fn zeros() -> Vec3 {
        Vec3(0., 0., 0.)
    }
    pub fn ones() -> Vec3 {
        Vec3(1., 1., 1.)
    }
}
impl Add<&Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}
impl Add<Vec3> for Vec3 {
    type Output = Vec3;

    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}
impl Sub<&Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}
impl Sub<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}
impl Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}
impl Sub<Vec3> for &Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}
impl Neg for &Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}
impl Mul<f32> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f32) -> Self::Output {
        Vec3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}
impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}
impl Mul<&Vec3> for &Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}
impl Div<f32> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f32) -> Self::Output {
        Vec3(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

#[derive(Debug, Clone)]
struct Sphere {
    weight: f32,
    rgb: RGB,

    radius: f32,

    center: Vec3,
    verosity: Vec3,
}

impl Sphere {
    fn weight(&self) -> f32 {
        self.weight
    }

    fn is_crossed(&self, rhs: &Sphere) -> bool {
        (self.center - &rhs.center).norm() <= (self.radius + rhs.radius)
    }

    fn progress(&self, dt: f32, force_field: fn(&Vec3) -> Vec3) -> Sphere {
        // Leap-frog
        let next_center = self.center + self.verosity * dt;
        let next_verocity = self.verosity + force_field(&next_center) * (dt / self.weight);

        Sphere {
            center: next_center,
            verosity: next_verocity,
            ..self.clone()
        }
    }
}

struct World {
    time_s: f32,
    objects: Vec<Sphere>,
    vector_field: fn(&Vec3) -> Vec3,
}

impl World {
    fn progress(&self) -> World {
        const DT: f32 = 1e-2;
        const EPS: f32 = 1. - 1e-2;

        let mut objects: Vec<Sphere> = self.objects.iter().map(|x| x.clone()).collect();
        for i in 1..objects.len() {
            let (lhs_list, rhs_list) = objects.split_at_mut(i);
            let rhs = &mut rhs_list[0];
            for j in 0..i {
                let lhs = &mut lhs_list[j];

                if lhs.is_crossed(rhs) {
                    let v = lhs.center - rhs.center;
                    let n = v / v.norm();
                    let lhs_vn = lhs.verosity.inner(&n);
                    let rhs_vn = rhs.verosity.inner(&n);

                    lhs.verosity = lhs.verosity + n * (rhs_vn - lhs_vn) * EPS;
                    rhs.verosity = rhs.verosity + n * (lhs_vn - rhs_vn) * EPS;
                }
            }
        }
        let objects: Vec<Sphere> = objects
            .iter()
            .map(|x| x.progress(DT, self.vector_field))
            .collect();

        World {
            time_s: self.time_s + DT,
            objects,
            vector_field: self.vector_field,
        }
    }

    fn time(&self) -> f32 {
        self.time_s
    }

    fn objects(&self) -> &[Sphere] {
        &self.objects
    }
}

use show_image::{create_window, event, ImageInfo, ImageView};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    const WIDTH: usize = 800;
    const HEIGHT: usize = 500;

    let world = World {
        time_s: 0.,
        objects: vec![
            Sphere {
                weight: 1.,
                rgb: RGB::blue(),
                radius: 1.,
                center: Vec3::zeros(),
                verosity: Vec3::ones() * 2.,
            },
            Sphere {
                weight: 1.,
                rgb: RGB::green(),
                radius: 0.6,
                center: Vec3::ones(),
                verosity: -Vec3::ones() * 5.,
            },
            Sphere {
                weight: 1.,
                rgb: RGB::red(),
                radius: 0.5,
                center: Vec3(-4., -1., 2.),
                verosity: -Vec3::ones(),
            },
            Sphere {
                weight: 1.,
                rgb: RGB::red(),
                radius: 0.5,
                center: Vec3(-4., 1., 2.),
                verosity: -Vec3::ones(),
            },
            Sphere {
                weight: 1.,
                rgb: RGB::red(),
                radius: 0.5,
                center: Vec3(-4., -1., -5.),
                verosity: -Vec3::ones(),
            },
            Sphere {
                weight: 1.,
                rgb: RGB::red(),
                radius: 1.,
                center: Vec3(-4., -9., -1.),
                verosity: -Vec3::ones(),
            },
            Sphere {
                weight: 1.,
                rgb: RGB::red(),
                radius: 1.,
                center: Vec3(1., -3., -6.),
                verosity: -Vec3::ones(),
            },
        ],
        vector_field: |x: &Vec3| -x * 5.,
    };
    let render = |world: &World, pixel_data: &mut [u8]| {
        const CAMERA_POSITION: Vec3 = Vec3(0., 0., 20.);
        const X_DIRECTION: Vec3 = Vec3(1., 0., 0.);
        const Y_DIRECTION: Vec3 = Vec3(0., 1., 0.);
        const Z_DIRECTION: Vec3 = Vec3(0., 0., -1.);

        const X_EYE_ANGLE: f32 = PI / 8.;
        const EYE_CELL_ANGLE: f32 = X_EYE_ANGLE / WIDTH as f32;
        const Y_EYE_ANGLE: f32 = EYE_CELL_ANGLE * HEIGHT as f32;
        const EYE_RADIUS: f32 = 0.3;
        let eye_center: Vec3 = CAMERA_POSITION + Z_DIRECTION * EYE_RADIUS;

        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let camera_pos = CAMERA_POSITION;
                let n = {
                    let x_angle = EYE_CELL_ANGLE * x as f32 - X_EYE_ANGLE / 2.;
                    let y_angle = EYE_CELL_ANGLE * y as f32 - Y_EYE_ANGLE / 2.;
                    let eye_pos =
                        eye_center + X_DIRECTION * x_angle.sin() + Y_DIRECTION * y_angle.sin();
                    let v = eye_pos - camera_pos;
                    v / v.norm()
                };

                let mut visible_obj: Option<(f32, &Sphere)> = None;
                for object in world.objects() {
                    let col_pos = {
                        let (e, d_e) = {
                            let v = object.center - camera_pos;
                            let dist = v.norm();
                            (v / dist, dist)
                        };

                        let cos = e.inner(&n);
                        let sin = (1. - cos.powi(2)).sqrt();
                        if cos >= 0. && (d_e * sin) <= object.radius {
                            let d_n =
                                d_e * cos - (object.radius.powi(2) - (d_e * sin).powi(2)).sqrt();
                            Some(camera_pos + n * d_n)
                        } else {
                            None
                        }
                    };
                    if let Some(col_pos) = col_pos {
                        let d = (col_pos - camera_pos).norm();
                        match visible_obj {
                            Some((min_dist, _)) if min_dist > d => {
                                visible_obj = Some((d, object));
                            }
                            None => {
                                visible_obj = Some((d, object));
                            }
                            _ => {}
                        }
                    }
                }

                let base: usize = (x + y * WIDTH) * 3;
                if let Some((d, obj)) = visible_obj {
                    let brawer = CAMERA_POSITION.norm().powi(2) / 1.4 / d.powi(2);
                    pixel_data[base + 0] = (obj.rgb.0 as f32 * brawer).max(1.).min(255.) as u8;
                    pixel_data[base + 1] = (obj.rgb.1 as f32 * brawer).max(1.).min(255.) as u8;
                    pixel_data[base + 2] = (obj.rgb.2 as f32 * brawer).max(1.).min(255.) as u8;
                } else {
                    pixel_data[base + 0] = 0;
                    pixel_data[base + 1] = 0;
                    pixel_data[base + 2] = 0;
                }
            }
        }
    };

    let world_history = {
        let mut tmp = vec![world];
        for i in 0..100000 {
            tmp.push(tmp[tmp.len() - 1].progress());
            println!("{}", i);
        }
        tmp
    };

    let window = create_window("image", Default::default())?;
    let mut pixel_data = vec![0_u8; WIDTH * HEIGHT * 3];
    for world in world_history {
        let start = Instant::now();

        render(&world, &mut pixel_data);
        let image = ImageView::new(ImageInfo::rgb8(WIDTH as u32, HEIGHT as u32), &pixel_data);
        window.set_image("image", image)?;

        let time = start.elapsed();

        if Duration::from_secs_f32(1e-2) > time {
            sleep((Duration::from_secs_f32(1e-2) - time).max(Duration::ZERO));
        }
    }

    // ESCキーが押されたら終了
    for event in window.event_channel()? {
        if let event::WindowEvent::KeyboardInput(event) = event {
            println!("{:#?}", event);
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }
    Ok(())
}
