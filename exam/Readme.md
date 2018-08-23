I have chosen to do project 28:
"
Rootfinding: 1D complex vs. 2D real

Implement a (quasi) Newton method for rootfinding for a complex function f of complex variable z,

	f(z) = 0.

Compare the effectiveness of your complex implementation with your homework multi-dimensional implementation of real rootfinding applied to the equivalent 2D system of two real equation with two real variables x and y,

	Re f(x + iy) = 0,
	Im f(x + iy) = 0.
"

For part A I have implemented a 1D complex quasi-Newton rootfinder using Broyden's method. I have tested the algorithm on three different functions (see out_A.txt).

For part B I have compared the 1D complex rootfinding algorithm with a 2D real-valued rootfinding algorithm. I find that the 1D complex rootfinding algorithm makes fewer function calls that its 2D real-valued counterpart.

For part C i have compared the effectiveness of three different rootfinding algorithms:

	1) quasi-Newton using Broyden's method
	2) quasi-Newton using symmetric rank-1 update
	3) Newtons method

I have implemented the three algorithms for both the 1D complex functions and the 2D real-valued functions.
