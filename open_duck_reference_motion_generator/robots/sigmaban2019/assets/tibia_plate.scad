% scale(1000) import("tibia_plate.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0, -67, 1.5]) {
    cube([30, 165, 3], center=true);
}
// cylinder(r=10, h=10, center=true);
// sphere(10);
