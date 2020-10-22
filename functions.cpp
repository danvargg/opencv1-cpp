/*
Created on: 2020-10-22
Author: @danvargg
*/

#include <stdio.h>
// #include "basic.cpp"

extern int reps = 0;   //Global variable
static int called = 0; //Local variable
const int limit = 10;  //Constant value

int square(int x)
{
    return (x * x);
}

// Functions
void very(int count)
{
    while (count > 0)
    {
        printf("\nvery");
        printf("\n%d", square(count));
        count--;
    };
    printf("much\n");
}

int main(void)
{
    int repeat = 10;
    auto int i = 1;
    very(repeat);
    return 0;
}