% scale(1000) import("radius_plate.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([-1.5, 42.5, 15]) {
  cube([3, 85, 30], center=true);
}

translate([9, 0, 15]) {
  cylinder(r=12, h=30, center=true);
}

// sphere(10);
