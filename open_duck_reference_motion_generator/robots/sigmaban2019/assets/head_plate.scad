% scale(1000) import("head_plate.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:

translate([0, 27, 1.5]) {
   for (x=[-37.5,0,37.5]) {
    translate([x,0,0]) {
      cube([8, 50, 3], center=true);
    }
  }
    
  for (y=[-23,0,23]) {
    translate([0,y,0]) {
      cube([83, 8, 3], center=true);
    }
  }
}

// cylinder(r=10, h=10, center=true);
// sphere(10);
