% scale(1000) import("foot_plate.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0, 0, 1.5]) {
  cube([90, 155, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
