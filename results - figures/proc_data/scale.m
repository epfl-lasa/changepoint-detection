 % Scale Variables 
 clear D; D = X;
 D(:,1:3)    = X(:,1:3)/1.5;
 D(:,4:7)    = X(:,4:7)/2;
 D(:,8:10)   = X(:,8:10)/15;
 D(:,11:end) = X(:,11:end)/2;
 X = D;