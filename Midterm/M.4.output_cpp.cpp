#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

using namespace std;

void forLoops(int,int);

int main()
{
   int N = 10,M = 80;            //Setting NxN to 10x10 and max index to 80
   forLoops(N,M);
   return 0;
}

void forLoops(int N, int M)
{
    int A[N][N], I[N][N], Mat[N][N];
	
    for (int i = 0;i<N;i++)
    {
        for (int j = 0;j<N;j++)
        {
            if (i == j)
			{
				I[i][j]=1;
			}
			else
			{
				I[i][j]=0;
			}
			
        }
    }
    for (int i = 0;i<N;i++ ) 
	{ 
        for (int j = 0;j<N;j++ ) 
		{
            cout<<I<<" ";
        }
    }
}



/*class Matrix
{
public:
	Matrix(); //Default constructor
	Matrix(int m, int n); //Main constructor
	void setVal(int m, int n); //Method to set the val of [i,j]th-entry
	void printMatrix(); //Method to display the matrix
	~Matrix(); //Destructor

private:
	int m, n;
	int **I, **A;
    int N = 10, max = 80;

	//allocate the array
	void allocArray()
	{
		I = new int*[m];
		for(int i = 0; i < m; i++)
		{
			I[i] = new int[n];
		}
	}
};

//Default constructor
Matrix::Matrix() : Matrix(1,1) {}

//Main construcor
Matrix::Matrix(int m, int n)
{
	allocArray();
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			if(i==j)
            {
                I[i][j] = 1;
                A[i][j] = 2/(N+1);
            }
            else
            {
                I[i][j] = 0;
                A[i][j] = 1/(N+1);
            }
		}
	}
    object.printA();
}


//destructor
Matrix::~Matrix()
{
	for(int i = 0; i < m; i++)
	{
		delete [] I[i];
	}
	delete [] p;
}

//SetVal function
void Matrix::setVal(int m, int n)
{
	int newVal = 0;
	p[m][n] = newVal;
}

//printMatrix function
void Matrix::printMatrix()
{
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			cout << p[m][n] << "  " << endl;
		}
	}
}


int main()
{
	int d1 = 4;
	int d2 = 3;

	Matrix object;

	//create 4x3 dynamic 2d array
	int ** matrix = new int*[d1];
	for(int j = 0; j < d1; j++)
	{
		matrix[j] = new int[d2];
	}

	//fill array
	cout << "Enter values " << endl;
	for(int i = 0; i < d1; ++i)
	{
		for(int j = 0; j < d2; ++j)
		{
			cin >> matrix[i][j];
		}
	}

	object.printMatrix();

	return 0;
}


int main()
{
    int N = 10;
    int max = 80;
    int i, j, 
    int A[N][N], I[N][N];

    for(i = 0; i < N; ++i)
    {
        for(j = 0; j < N; ++j)
        {
            
            if(i == j)
            {
                A[i][j] = 2/(N+1);
                I[i][j] = 1;
                }
            else
            {
                A[i][j] = 1/(N+1);
                I[i][j] = 0;
            }
            
        }
    }
    return 0;
}
cout << A*/