% scale(1000) import("nuc_fix.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([-7, 0, 1.5]) {
    cube([10, 135, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
