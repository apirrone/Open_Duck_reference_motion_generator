% scale(1000) import("u_side_femur_mir.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0, 46, 44.5]) {
  cube([40, 90, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
