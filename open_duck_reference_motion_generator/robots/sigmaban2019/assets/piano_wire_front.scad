% scale(1000) import("piano_wire_front.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
translate([0, 45, 0]) {
    cylinder(r=40, h=3, center=true);
}
// sphere(10);
