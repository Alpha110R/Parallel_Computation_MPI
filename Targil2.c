#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include<stdbool.h>
#include "Targil2.h"
#define FILE_NAME "rectangles.dat"
#define MASTER 0
#define COMM_DIMENSION 2
#define TAG 0

int main(int argc, char* argv[]) {  
    int numberOfPoints,
        my_rank,
        num_procs,
        numberOfRectangles;
    int gridDimension[2];
    double myRectangle[3], readyToSendToGatherFinish[3];
    double arrIdRectanleArea[2];
    double* rectangles;//MASTER
    double* gatherArray;
    double myRectangleArea,//[0]: ID [1]: Rectangle area
           rowsColumns;
    FILE* end_file;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if(my_rank == MASTER){
        end_file = fopen("result.dat","w");
        if(!end_file){
            printf("Error writing to file");
        }
        rectangles = readFromFile(FILE_NAME, &numberOfRectangles);
        if (num_procs != numberOfRectangles) {
            fprintf(stderr, "number of processes should be %d\n", numberOfRectangles);
            MPI_Abort(MPI_COMM_WORLD, 1); 
        }
        double result = sqrt((double)numberOfRectangles);
        gridDimension[0] = result;
        gridDimension[1] = result;
    }
    MPI_Scatter(rectangles, 3, MPI_DOUBLE, myRectangle, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gridDimension, 2, MPI_INT, MASTER, MPI_COMM_WORLD);
    arrIdRectanleArea[1] = calcRectangleArea(myRectangle[1], myRectangle[2]);
    arrIdRectanleArea[0] = myRectangle[0];
    buildNewCommAndSortIt(gridDimension, my_rank, arrIdRectanleArea);
    if(my_rank == MASTER)
        gatherArray =(double*)malloc(3 * numberOfRectangles * sizeof(double));
    readyToSendToGatherFinish[0] = my_rank;
    readyToSendToGatherFinish[1] = arrIdRectanleArea[0];
    readyToSendToGatherFinish[2] = arrIdRectanleArea[1];
    MPI_Gather(readyToSendToGatherFinish, 3, MPI_DOUBLE, gatherArray, 3, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    if(my_rank == MASTER)
        printSnakeIdRectangle(gatherArray, gridDimension[0], end_file);
    MPI_Finalize();
    return 0;
}
/* Each process has two coordinates.
       We use the convention that the first coordinate is the row number,
       The second coordinate is the column number
*/
void buildNewCommAndSortIt(int* gridDimension, int rank, double* arrIdRectanleArea){
    int periods[COMM_DIMENSION] = {1, 1};
    int reorderRank = 0;
    MPI_Comm newComm;
    int coordinateOfProces[COMM_DIMENSION];
    if(MPI_Cart_create(MPI_COMM_WORLD, 2, gridDimension, periods, reorderRank, &newComm) != MPI_SUCCESS){
        fprintf(stderr, "Create Cartesian Topology failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Cart_coords(newComm, rank, COMM_DIMENSION, coordinateOfProces);  
    shearSort(gridDimension, rank, arrIdRectanleArea, coordinateOfProces, newComm);
}

void shearSort(int* gridDimension, int rank, double* arrIdRectanleArea, int* coordinateOfProces, MPI_Comm newComm){
    int number_of_iterations = (gridDimension[1]+1)/2;//number of iteration rounded up
    for (int i = 0; i <number_of_iterations+1; i++) {
        sortRows(gridDimension, rank, arrIdRectanleArea, coordinateOfProces, newComm, number_of_iterations);
        sortColumns(gridDimension, rank, arrIdRectanleArea, coordinateOfProces, newComm, number_of_iterations);
    }
}

void sortColumns(int* gridDimension, int rank, double* arrIdRectanleArea, int* coordinateOfProces, MPI_Comm newComm, int number_of_iterations){
    int rankNeighborUp,//left
        rankNeighborDown;//Right
    MPI_Cart_shift(newComm, 0, 1, &rankNeighborUp, &rankNeighborDown);
    for (int i = 0; i <number_of_iterations+1; i++) {
        int neighbour,
            flagToSwitch=-1,
            flagToSwitchNeighbor=-1;
        double rectangleAreaNeighbor;
        if (coordinateOfProces[0] % 2 == 0)
            neighbour = (i % 2 == 0) ? rankNeighborDown : rankNeighborUp;
        else 
            neighbour = (i % 2 == 0) ? rankNeighborUp : rankNeighborDown;
        MPI_Sendrecv(&arrIdRectanleArea[1], 1, MPI_DOUBLE, neighbour, TAG,
                     &rectangleAreaNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
                    
        if(rectangleAreaNeighbor <= arrIdRectanleArea[1])               
            MinToMaxSort(&flagToSwitch, neighbour, newComm, rank, rectangleAreaNeighbor, arrIdRectanleArea);
        else
            MPI_Recv(&flagToSwitchNeighbor, 1, MPI_INT , neighbour, TAG, newComm , MPI_STATUS_IGNORE);   
        if(flagToSwitchNeighbor==1 || (flagToSwitch==1))
            switchAreasBetweenProcess(arrIdRectanleArea, neighbour, newComm, rank);
    }
}

void sortRows(int* gridDimension, int rank, double* arrIdRectanleArea, int* coordinateOfProces, MPI_Comm newComm, int number_of_iterations){
    int rankNeighborLeft,
        rankNeighborRight;
    MPI_Cart_shift(newComm, 1, 1, &rankNeighborLeft, &rankNeighborRight);
    for (int i = 0; i <number_of_iterations+1; i++) {
        int neighbour,
            flagToSwitch=-1,
            flagToSwitchNeighbor=-1;
        double rectangleAreaNeighbor;
        if (coordinateOfProces[1] % 2 == 0)
            neighbour = (i % 2 == 0) ? rankNeighborRight : rankNeighborLeft;
        else 
            neighbour = (i % 2 == 0) ? rankNeighborLeft : rankNeighborRight;
        MPI_Sendrecv(&arrIdRectanleArea[1], 1, MPI_DOUBLE, neighbour, TAG,
                     &rectangleAreaNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
                    
        if(rectangleAreaNeighbor <= arrIdRectanleArea[1]){               
            procesNeedToCheckSwitch(&flagToSwitch, neighbour, newComm, rank,coordinateOfProces, rectangleAreaNeighbor, arrIdRectanleArea);
        }
        else
            MPI_Recv(&flagToSwitchNeighbor, 1, MPI_INT , neighbour, TAG, newComm , MPI_STATUS_IGNORE);   
        if(flagToSwitchNeighbor==1 || (flagToSwitch==1))
            switchAreasBetweenProcess(arrIdRectanleArea, neighbour, newComm, rank);
    }
}

void procesNeedToCheckSwitch(int* flagToSwitch, int neighbour, MPI_Comm newComm, int rank, int* coordinateOfProces, double rectangleAreaNeighbor, double* arrIdRectanleArea){
    if(coordinateOfProces[0]%2 ==0){
        MinToMaxSort(flagToSwitch, neighbour, newComm, rank, rectangleAreaNeighbor, arrIdRectanleArea);
    }else{
        MaxToMinSort(flagToSwitch, neighbour, newComm, rank, rectangleAreaNeighbor, arrIdRectanleArea);
    }
}

void MinToMaxSort(int* flagToSwitch, int neighbour, MPI_Comm newComm, int rank, double rectangleAreaNeighbor, double* arrIdRectanleArea){
    if(rectangleAreaNeighbor < arrIdRectanleArea[1])
        switchAreasRowsMinToMax(flagToSwitch, neighbour, newComm, rank);
    else{
        if(rank > neighbour)
            switchAreasRowsMinToMaxByID(neighbour, newComm, arrIdRectanleArea, rank);
        else{
            MPI_Send(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm);
            MPI_Recv(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
        }         
    }
}

void MaxToMinSort(int* flagToSwitch, int neighbour, MPI_Comm newComm, int rank, double rectangleAreaNeighbor, double* arrIdRectanleArea){
    if(rectangleAreaNeighbor < arrIdRectanleArea[1])
        switchAreasRowsMaxToMin(flagToSwitch, neighbour, newComm, rank);
    else{
        if(rank > neighbour)
            switchAreasRowsMaxToMinByID(neighbour, newComm, arrIdRectanleArea, rank);
        else{
            MPI_Send(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm);
            MPI_Recv(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
        }         
    }
}

void switchAreasRowsMinToMax(int* flagToSwitch, int neighbour, MPI_Comm newComm, int rank){
    if(neighbour < rank)
        *flagToSwitch =0;//flagToSwitch =0 -> dont switch
    else
        *flagToSwitch =1;
    MPI_Send(flagToSwitch , 1, MPI_INT , neighbour, TAG, newComm);
}

void switchAreasRowsMaxToMin(int* flagToSwitch, int neighbour, MPI_Comm newComm, int rank){
    if(neighbour > rank)
        *flagToSwitch =0;//flagToSwitch =0 -> dont switch
    else
        *flagToSwitch =1;
    MPI_Send(flagToSwitch , 1, MPI_INT , neighbour, TAG, newComm);
}

void switchAreasBetweenProcess(double* arrIdRectanleArea, int neighbour, MPI_Comm newComm, int rank){
    double tempRectangleAreaFromNeighbour;
    MPI_Sendrecv(arrIdRectanleArea, 2, MPI_DOUBLE, neighbour, TAG,
                arrIdRectanleArea, 2, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
               // arrIdRectanleArea[1] = tempRectangleAreaFromNeighbour;
}

void switchAreasRowsMinToMaxByID(int neighbour, MPI_Comm newComm, double* arrIdRectanleArea, int rank){
    double rectangleIDNeighbor,
           myRectangleID;
    myRectangleID = arrIdRectanleArea[0];
    MPI_Recv(&rectangleIDNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
    if(myRectangleID < rectangleIDNeighbor){
        MPI_Send(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm);
        arrIdRectanleArea[0] = rectangleIDNeighbor;
    }
    else
        MPI_Send(&rectangleIDNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm);

}
void switchAreasRowsMaxToMinByID(int neighbour, MPI_Comm newComm, double* arrIdRectanleArea, int rank){
    double rectangleIDNeighbor,
           myRectangleID;
    myRectangleID = arrIdRectanleArea[0];
    MPI_Recv(&rectangleIDNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm, MPI_STATUS_IGNORE);
    if(myRectangleID > rectangleIDNeighbor){
        MPI_Send(&arrIdRectanleArea[0], 1, MPI_DOUBLE, neighbour, TAG, newComm);
        arrIdRectanleArea[0] = rectangleIDNeighbor;
    }
    else
        MPI_Send(&rectangleIDNeighbor, 1, MPI_DOUBLE, neighbour, TAG, newComm);
}

// Reads a number of rectangles from the file.
// The first line contains a number of rectangles defined.
// Following lines contain three doubles: ID side1 side2
double *readFromFile(const char *fileName, int* numberOfRectangles) {
    FILE* fp;
    double* rectangles;
    if ((fp = fopen(fileName, "r")) == 0) {
        printf("cannot open file %s for reading\n", fileName);
        exit(0);
    }
    fscanf(fp, "%d",numberOfRectangles);
    rectangles = (double*)malloc(3 * *numberOfRectangles * sizeof(double));
    if (rectangles == NULL) {
        printf("Problem to allocate memotry\n");

        exit(0);
    }
    for (int i = 0; i < *numberOfRectangles; i++) {
        fscanf(fp, "%lf %lf %lf", &rectangles[3*i], &rectangles[3*i + 1], &rectangles[3*i + 2]);
    }
    fclose(fp);
    return rectangles;
}

double calcRectangleArea(double side1, double side2){
    return side1*side2;
}

void printSnakeIdRectangle(double *arr, int sizeSide, FILE* file){
	int count = 0;
	int count2 = 0;
	for(int i = 0 ; i < 3 * (sizeSide *sizeSide); i += 3){
		if( count % 2 == 1){
			for( int j = i + (sizeSide * 3); j > i ; j-=3){
				printf("Proces: %lf  Rectangle ID: %lf   Rectangle area: %lf\n", arr[j-3], arr[j-2], arr[j-1]);
				 fprintf(file,"Rectangle ID: %d\n", (int)arr[j-2]);
			}
			i += (sizeSide * 3) -  3;
			count2 = 0;
			count += 1;
		}
		else{
			printf("Proces: %lf  Rectangle ID: %lf   Rectangle area: %lf\n", arr[i], arr[i+1], arr[i+2]);
            fprintf(file,"Rectangle ID: %d\n", (int)arr[i+1]);
			count2+= 1;
		}
		if(count2 == 4){
			count += 1;
		}
	}
	fclose(file);
}









