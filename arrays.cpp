/*
Created on: 2020-10-22
Author: @danvargg
*/

#include <stdio.h>

int main(void)
{
    const int SIZE = 5;
    int grades[SIZE] = {35, 21, 62, 34, 47};
    double sum = 0.0;
    double *ptr_to_sum = &sum; //Pointer declaration
    int i;

    printf("Grades: \n");

    for (i = 0; i < SIZE; i++)
        printf("%d\t", grades[i]);

    printf("\n\n");

    for (i = 0; i < SIZE; i++)
        sum = sum + grades[i];

    printf("Average: ");

    printf("%.2f\n", sum / SIZE);

    printf("%p\t%lf\n", ptr_to_sum, *ptr_to_sum);

    return 0;
}