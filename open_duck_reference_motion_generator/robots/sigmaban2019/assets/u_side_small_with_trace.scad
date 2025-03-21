% scale(1000) import("u_side_small_with_trace.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([15, 23, 1.5]) {
 cube([27, 42, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
