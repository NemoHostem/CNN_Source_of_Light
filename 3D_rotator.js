// # -*- coding: utf-8 -*-

/*
Created on Wed Jun 19 10:01:41 2019

@authors: Peter Collingridge, Matias Ijäs
*/

/*
See tutorial at:
http://www.petercollingridge.appspot.com/3D-tutorial
3D tutorial 2 and browser test environment:
https://www.khanacademy.org/computer-programming/3d-tutorial-2/1645140418
More theory about rotations:
http://petercollingridge.appspot.com/3D-tutorial/rotating-objects
*/


var backgroundColour = color(255, 255, 255);
var nodeColour = color(40, 168, 107);
var edgeColour = color(34, 68, 204);
var nodeSize = 8;

var node0 = [-100, -150, -80];
var node1 = [-100, -150,  20];
var node2 = [-100,  150, -80];
var node3 = [-100,  150,  20];
var node4 = [ 100, -150, -80];
var node5 = [ 100, -150,  20];
var node6 = [ 100,  150, -80];
var node7 = [ 100,  150,  20];
var node8 = [ 0,  0,  0];
var node9 = [ 0,  0,  -200];
var nodes = [node0, node1, node2, node3, node4, node5, node6, node7, node8, node9];

var edge0  = [0, 1];
var edge1  = [1, 3];
var edge2  = [3, 2];
var edge3  = [2, 0];
var edge4  = [4, 5];
var edge5  = [5, 7];
var edge6  = [7, 6];
var edge7  = [6, 4];
var edge8  = [0, 4];
var edge9  = [1, 5];
var edge10 = [2, 6];
var edge11 = [3, 7];
var edge12 = [0, 8];
var edge13 = [2, 8];
var edge14 = [4, 8];
var edge15 = [6, 8];
var edge16 = [8, 9];
var edges = [edge0, edge1, edge2, edge3, edge4, edge5, edge6, edge7, edge8, edge9, edge10, edge11, edge12, edge13, edge14, edge15, edge16];

var rotateX3D = function(theta) {
    var sin_t = sin(theta);
    var cos_t = cos(theta);
    
    for (var n = 0; n < nodes.length; n++) {
        var node = nodes[n];
        var y = node[1];
        var z = node[2];
        node[1] = y * cos_t - z * sin_t;
        node[2] = z * cos_t + y * sin_t;
    }
};

var rotateY3D = function(theta) {
    var sin_t = sin(theta);
    var cos_t = cos(theta);
    
    for (var n = 0; n < nodes.length; n++) {
        var node = nodes[n];
        var x = node[0];
        var z = node[2];
        node[0] = x * cos_t - z * sin_t;
        node[2] = z * cos_t + x * sin_t;
    }
};

// Rotate shape around the z-axis
var rotateZ3D = function(theta) {
    var sin_t = sin(theta);
    var cos_t = cos(theta);
    
    for (var n=0; n<nodes.length; n++) {
        var node = nodes[n];
        var x = node[0];
        var y = node[1];
        node[0] = x * cos_t - y * sin_t;
        node[1] = y * cos_t + x * sin_t;
    }
};

var draw= function() {
    background(backgroundColour);
    
    // Draw edges
    stroke(edgeColour);
    for (var e=0; e<edges.length-1; e++) {
        var n0 = edges[e][0];
        var n1 = edges[e][1];
        var node0 = nodes[n0];
        var node1 = nodes[n1];
        line(node0[0], node0[1], node1[0], node1[1]);
    }
    stroke(255,0,255);
    e = edges.length-1;
    var n0 = edges[e][0];
    var n1 = edges[e][1];
    var node0 = nodes[n0];
    var node1 = nodes[n1];
    line(node0[0], node0[1], node1[0], node1[1]);
    
    // Draw nodes
    fill(nodeColour);
    noStroke();
    for (var n=0; n<nodes.length - 1; n++) {
        var node = nodes[n];
        ellipse(node[0], node[1], nodeSize, nodeSize);
    }
    fill(255,0,0);
    noStroke();
    var node = nodes[nodes.length - 1];
    ellipse(node[0], node[1], nodeSize, nodeSize);
    
};

translate(200, 200);

var mouseDragged = function() {
    rotateY3D(mouseX - pmouseX);
    rotateX3D(mouseY - pmouseY);
};

var mousePressed = function() {
    if (mouseIsPressed && mouseButton === LEFT) {
        rotateZ3D(-5);
    } else if (mouseIsPressed && mouseButton === RIGHT) {
        rotateZ3D(5);
    }
    debug(node9);
    debug(-1*atan2(node9[1],node9[0]));
};

rotateX3D(5);
rotateY3D(0);
rotateZ3D(-80);
debug(node9);
debug(-1*atan2(node9[1],node9[0]));
