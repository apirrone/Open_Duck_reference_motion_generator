% scale(1000) import("torso_bottom_plate_nuc.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0,23, 1.5]) {
  cube([130, 44, 3], center=true);
}

translate([-30, -16, 1.5]) {
  cube([10, 35, 3], center=true);
}

translate([30, -16, 1.5]) {
  cube([10, 35, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
