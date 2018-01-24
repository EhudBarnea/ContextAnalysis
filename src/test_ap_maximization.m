
tmp=[];
mx = -1;
mn = 1;
mxv=0;
mnv=0;
c1 = 0;
c2 = 0;
while true
    nn = 1000;
    x0=randi(nn);
    x1=randi(nn);
    x2=randi(nn);
    y0=randi(nn);
    y1=randi(nn);
    y2=randi(nn);
    
    p0 = x0/(x0+y0);
    p1 = x1/(x1+y1);
    p2 = x2/(x2+y2);
    if p1 <= p2
        continue;
    end
    if p0 <= p1
        continue;
    end
    
    v=[[x0,y0,p0];[x1,y1,p1];[x2,y2,p2]];
    
    pos = (x0+x1+x2)*1;
    ap1 = (x0/pos)*(x0)/(x0+y0) + (x1/pos)*(x0+x1)/(x0+x1+y0+y1) + (x2/pos)*(x0+x1+x2)/(x0+x1+x2+y0+y1+y2);
    tmpx=x1;
    tmpy=y1;
    x1=x2;
    y1=y2;
    x2=tmpx;
    y2=tmpy;
    ap2 = (x0/pos)*(x0)/(x0+y0) + (x1/pos)*(x0+x1)/(x0+x1+y0+y1) + (x2/pos)*(x0+x1+x2)/(x0+x1+x2+y0+y1+y2);
    
    tmpx=x1;
    tmpy=y1;
    x1=x2;
    y1=y2;
    x2=tmpx;
    y2=tmpy;
    if ap1>ap2
        c1=c1+1;
    else
        c2=c2+1;
    end
    if mn > ap1-ap2
        mn = ap1-ap2;
        mnv = v;
        mn
        v
    end
end

% when p2 is smaller but very close to p1 and x2 is several times smaller
% than x1. everything is good if x2>x1.
% an example of the smallest ap1-ap2 found is about ap1-ap2=0.6093-0.6257=-0.0164 with:
%   277.0000   16.0000    0.9454
%   371.0000  955.0000    0.2798
%    69.0000  178.0000    0.2794

%%

nn = 1000;
x=[0,0,0];
y=[0,0,0];
x(1)=randi(nn);
x(2)=randi(nn);
x(3)=randi(nn);
y(1)=randi(nn);
y(2)=randi(nn);
y(3)=randi(nn);

v=[x', y', (x./(x+y))'];
pos = (x0+x1+x2)*1;
taken = [0,0,0];
order = [0,0,0];

for i=1:3
    elvals = [0,0,0];
    for el=1:3
        if taken(el)
            continue;
        end
        up = 0;
        down = 0;
        for j=1:i-1
            up = up + x(order(j));
            down = down + x(order(j)) + y(order(j));
        end
        
        elvals(el) = (x(el)/pos)*((up+x(el))/(down+x(el)+y(el)));
    end
    
    [~,mm] = max(elvals);
    order(i) = mm;
    taken(mm) = 1;
end

v(order,:)

%%

x0=10;
x1=10;
x2=20;
y0=10;
y1=10;
y2=11;

scores = [0,0; 0,0];

pos = x0+x1+x2;
% pos = 1;
ap = (x0/pos)*(x0)/(x0+y0) + (x1/pos)*(x0+x1)/(x0+x1+y0+y1) + (x2/pos)*(x0+x1+x2)/(x0+x1+x2+y0+y1+y2);
test = x1*(x0+x1)/(x0+x1+y0+y1)+(x0+x1+x2)*(x2-x1)/(x0+y0+x1+y1+x2+y2)-x2*(x0+x2)/(x0+x2+y0+y2);
scores(1,1) = ap;
scores(2,1) = test;

tmpx=x1;
tmpy=y1;
x1=x2;
y1=y2;
x2=tmpx;
y2=tmpy;
ap = (x0/pos)*(x0)/(x0+y0) + (x1/pos)*(x0+x1)/(x0+x1+y0+y1) + (x2/pos)*(x0+x1+x2)/(x0+x1+x2+y0+y1+y2);
test = x1*(x0+x1)/(x0+x1+y0+y1)+(x0+x1+x2)*(x2-x1)/(x0+y0+x1+y1+x2+y2)-x2*(x0+x2)/(x0+x2+y0+y2);
scores(1,2) = ap;
scores(2,2) = test;


scores

