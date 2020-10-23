%Midterm Problem #4 (matlab) - Jack James

%ln(A)=-summation((I-A)^k/k,k=1,infinity)

N = 10;
max = 100;                            %Maximum k we will execute. This value is large enough that precision isn't improved by increasing it
                                      %since only 9 digits are displayed. 
                                    
A = zeros(N,N);

for i = 1:N                           %loop to create requested Aij=(1 + Kroniker Delta ij)/(n+1)
    for j = 1:N
        if i == j
            A(i,j) = 2/(N+1);
        
        else
            A(i,j) = 1/(N+1);
        end
    end
end

%disp(A)

I = zeros(N,N);

for k = 1:N                          %loop to create identity matrix
    for l = 1:N
        if k == l
            I(k,l) = 1;
        end
    end
end

%disp(I)

Eddie = log_mat(A,N,max,I);                               %This is the requested result
disp(Eddie)

%fid = fopen('C:/Users/Jack/Documents/Fall 2020/Comp Phys/Assignments/Midterm/M.4.output.txt','w');
%fprintf('fid = %f');
%fclose(fid);

writematrix(Eddie)
type 'C:/Users/Jack/Documents/Fall 2020/Comp Phys/Assignments/Midterm/M.4.output.txt'



function [VanHalen] = log_mat(A,N,max,I)                 %creating function that sums up taylor series given
    VanHalen = zeros(N,N);                               %VanHalen is an empty matrix we will add each series term to. 
    %Alex = I-A;
    for k = 1:(max+1)                                    %I get different results with python vs matlab...
        %Roth = Alex^k;                                  %I verified A's and I's were the same.
        %Hagar = Roth/k;                                 %fixed python
        %VanHalen = VanHalen - Hagar;
        VanHalen = VanHalen - ((I-A)^k)/k; 
    end
end

%PleaseMatch = test(A,N,max,I);

%function [answer] = test(A,N,max,I)
%   syms k
%   answer = zeros(N,N);
%   answer = answer-symsum(((I-A)^k/k),k,1,max);
%end
%original_stdout = sys.stdout

%with open('C:/Users/Jack/Documents/Fall 2020/Comp Phys/Assignments/Midterm/M.4.output.txt', 'w') as f:    %exporting to file
%    sys.stdout = f
%    #print("file")
%    print(log_mat(A,N,max))
%    sys.stdout = original_stdout