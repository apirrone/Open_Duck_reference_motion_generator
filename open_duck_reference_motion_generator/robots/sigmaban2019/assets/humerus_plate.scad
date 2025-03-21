% scale(1000) import("humerus_plate.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([1.5, 35, 15]) {
  cube([3, 70, 30], center=true);
}

translate([17.5, 73, 15]) {
  cube([30, 3, 30], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
