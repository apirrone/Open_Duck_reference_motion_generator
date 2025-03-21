% scale(1000) import("torso_top_plate_nuc.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0, -12, 1.5]) {
    cube([130, 80, 3], center=true);
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
