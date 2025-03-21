% scale(1000) import("handle.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([-45, 0, 1.5]) {
  cube([10, 130, 3], center=true);
}

for (y=[-60,60]) {
    translate([-20, y, 1.5]) {
      cube([40, 10, 3], center=true);
    }
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
