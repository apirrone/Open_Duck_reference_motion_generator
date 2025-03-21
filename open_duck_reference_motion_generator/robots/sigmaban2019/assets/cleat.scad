% scale(1000) import("cleat.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
// sphere(10);

rotate([90, 0, 0]) {
    translate([0, 0, -10]) {
        cylinder(r=8, h=20, center=true);
    }
}
