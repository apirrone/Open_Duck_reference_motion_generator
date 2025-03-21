% scale(1000) import("pillar_tube.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);

translate([0, 0, 43]) {
  cylinder(r=4, h=86, center=true);
}
// sphere(10);
