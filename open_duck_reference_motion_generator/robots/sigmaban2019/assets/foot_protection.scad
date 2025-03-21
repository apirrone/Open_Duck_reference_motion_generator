% scale(1000) import("foot_protection.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([-19, -27.5, 12.5]) {
rotate([0, 45, 0]) {
  cube([10, 90, 10], center=true);
}
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
