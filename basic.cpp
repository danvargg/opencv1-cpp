/*
Created on: 2020-10-19
Author: @danvargg
*/

#include <stdio.h>

// Another type of comment
// '\a' makes a sound

int main(void)
{
    float x, y, sum;
    printf("Input two floats:");
    scanf("%f%f", &x, &y);
    printf("x = %f, y = %f\n", x, y);
    sum = x + y;
    printf("sum = %.2f\n\n", sum);

    // Other stuff
    int a = 5, b = 7, c = 0, d = 0;
    c = a - b;
    printf("a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);
    c = ++a + b++;
    d += 5;
    printf("a = %d, b = %d, c = %d, d = %d\n", a, b, c, d);
    printf("d size: %lu\n", sizeof(d));
    return 0;
}
