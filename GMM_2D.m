
%% read file & RGB layer
image = imread("Shiba_1.jpg");
copy = image;
img_gray = rgb2gray(image);
img_gray = double(img_gray);
img = double(image);
[m,n, p] = size(img);
num = m*n;
imgR = img(:,:,1); % R
imgG = img(:,:,2); % G
imgB = img(:,:,3); % B
data = zeros(num,3);
%% Normalization
imgR = imgR / 255;
imgG = imgG / 255;
imgB = imgB / 255;
data(:,1) = imgR(:);
data(:,2) = imgG(:);
data(:,3) = imgB(:);
k=3;
shiba1_3D_k3 = threeD(data, k, m, n);
shiba1_1D_k3 = oneD(img_gray, k, copy);
%imwrite(shiba1_3D_k3, "output/shiba1_3D_k3.jpg");
%imwrite(shiba1_1D_k3, "output/shiba1_1D_k3.jpg");
k=5;
shiba1_3D_k5 = threeD(data, k, m, n);
shiba1_1D_k5 = oneD(img_gray, k, copy);
%imwrite(shiba1_3D_k5, "output/shiba1_3D_k5.jpg");
%imwrite(shiba1_1D_k5, "output/shiba1_1D_k5.jpg");
k=10;
shiba1_3D_k10 = threeD(data, k, m, n);
shiba1_1D_k10 = oneD(img_gray, k, copy);
%imwrite(shiba1_3D_k10, "output/shiba1_3D_k10.jpg");
%imwrite(shiba1_1D_k10, "output/shiba1_1D_k10.jpg");

image = imread("Shiba2.jpg");
copy = image;
img_gray = rgb2gray(image);
img_gray = double(img_gray);
img = double(image);
[m,n, p] = size(img);
num = m*n;
imgR = img(:,:,1); % R
imgG = img(:,:,2); % G
imgB = img(:,:,3); % B
data = zeros(num,3);
%% Normalization
imgR = imgR / 255;
imgG = imgG / 255;
imgB = imgB / 255;
data(:,1) = imgR(:);
data(:,2) = imgG(:);
data(:,3) = imgB(:);
k=3;
shiba2_3D_k3 = threeD(data, k, m, n);
shiba2_1D_k3 = oneD(img_gray, k, copy);
%imwrite(shiba2_3D_k3, "output/shiba2_3D_k3.jpg");
%(shiba2_1D_k3, "output/shiba2_1D_k3.jpg");
k=5;
shiba2_3D_k5 = threeD(data, k, m, n);
shiba2_1D_k5 = oneD(img_gray, k, copy);
%imwrite(shiba2_3D_k5, "output/shiba2_3D_k5.jpg");
%imwrite(shiba2_1D_k5, "output/shiba2_1D_k5.jpg");
k=10;
shiba2_3D_k10 = threeD(data, k, m, n);
shiba2_1D_k10 = oneD(img_gray, k, copy);
%imwrite(shiba2_3D_k10, "output/shiba2_3D_k10.jpg");
%imwrite(shiba2_1D_k10, "output/shiba2_1D_k10.jpg");

subplot(4, 3, 1); title('shiba1_1D_k3'); imshow(shiba1_1D_k3);
subplot(4, 3, 2); title('shiba1_1D_k5'); imshow(shiba1_1D_k5);
subplot(4, 3, 3); title('shiba1_1D_k10'); imshow(shiba1_1D_k10);
subplot(4, 3, 4); title('shiba1_3D_k3'); imshow(shiba1_3D_k3);
subplot(4, 3, 5); title('shiba1_3D_k5'); imshow(shiba1_3D_k5);
subplot(4, 3, 6); title('shiba1_3D_k10'); imshow(shiba1_3D_k10);
subplot(4, 3, 7); title('shiba2_1D_k3'); imshow(shiba2_1D_k3);
subplot(4, 3, 8); title('shiba2_1D_k5'); imshow(shiba2_1D_k5);
subplot(4, 3, 9); title('shiba2_1D_k10'); imshow(shiba2_1D_k10);
subplot(4, 3, 10); title('shiba2_3D_k3'); imshow(shiba2_3D_k3);
subplot(4, 3, 11); title('shiba2_3D_k5'); imshow(shiba2_3D_k5);
subplot(4, 3, 12); title('shiba2_3D_k10'); imshow(shiba2_3D_k10);

% imshow(output_1D_k10)
function [imgOutput] = threeD(data, k, m, n)
[pdf, color, v]= em_gaussian_3D(data, k);
imgReshape = zeros(n, 3);
kColor = jet(k);
kColor = color(:, 1:3);
imgReshape = pdf * kColor;
imgOutput = zeros(m, n, 3);
for i = 1:3
    imgOutput(:, :, i) = reshape(imgReshape(:,i), [m,n]);
end
% imshow(imgOutput);
end

function [output] = oneD(img_gray, k, copy)
[pdf]= em_gaussian_1D(img_gray, k);
%先定義k種顏色
randcolor = zeros(k, 3);
for i = 1:k
    randcolor(i, :) = [randi(255) randi(255) randi(255)];
end
[s1 s2]=size(img_gray);
%tmp = zeros(s1*s2, 1);
for i=1 : s1*s2
        c = pdf(i, :);
        a=find(c==max(c));  
        tmp(i, 1) = a(1);
end
mask = reshape(tmp, [s1 s2]);
output = copy;
for i = 1:s1
    for j = 1:s2
        ctmp = randcolor(mask(i,j), :);
        output(i,j, 1) = ctmp(1);%randcolor(mask(i,j), :);
        output(i,j, 2) = ctmp(2);
        output(i,j, 3) = ctmp(3);
    end
end
%imshow(output);
end
%% initial weight & probability matrix

%% function
function [pdf] = em_gaussian_1D(data, k)
    X = data(:);
    m = size(X, 1);
    indeces = randperm(m); % 隨機更換順序
    mu = zeros(1, k);
    for i = 1:k
        mu(i) = X(indeces(i));
    end
    sigma = ones(1, k) * sqrt(var(X));
    phi = ones(1,k)*1/k;
    W = zeros(m,k);
    
    for iter = 1:1000
        fprintf('  EM Iteration %d\n', iter);
        pdf = zeros(m, k);
        %E
        for j = 1 : k
            pdf(:, j) = gaussian1D(X, mu(j), sigma(j));
            %meanDiff = bsxfun(@minus, X, mu(j, :));
            %pdf(:, j) =1 / sqrt((2*pi) * det(sigma{j})) * exp(-1/2 * sum((meanDiff * inv(sigma{j}) .* meanDiff), 2));
        end
        pdf_w = bsxfun(@times, pdf, phi);
        W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2));
        %M
        prevMu = mu;
        prevSig =sigma;
        for j = 1 : k
            phi(j) = mean(W(:, j));     
            mu(j) = weightedAverage(W(:, j), X); 
            variance = weightedAverage(W(:, j), (X - mu(j)).^2);
            sigma(j) = sqrt(variance);
        end
        
%         if ((mu-prevMu) < 10^(-3))
        if sum((mu-prevMu).^2) < 1
            break
        end
    end
end

function [pdf, color, v] = em_gaussian_3D(data, k)
[n, dim] = size(data);
color = data(randi([1, n], k, 1), :); % k個random color
v = zeros(k, 1);

for i = 1:k
    tmp = data(i:k:end, 1);
    v(i, :) = std(tmp);
end

w = ones(k, 1)/k;
pdf = zeros(n, k);

c_0 = color * 0;
v_0 = 0 * v;
w_0 = w * 0;
x_u = zeros(size(data));
energy=sum(sum((color-c_0).^2))+sum(sum((v-v_0).^2))+(sum((w-w_0).^2));
iter = 1;
while energy > 10^(-6) % 取變化很小時為收斂
    fprintf('Iteration %d\n', iter);
    %% E
    for j = 1:k
        for l = 1:dim
            x_u(:, l) = data(:, l) - color(j, l) * ones(n , 1);
        end
        x_u = x_u.*x_u;
        pdf(:, j) = power(sqrt(2*pi)*v(j),-1*dim)*exp((-1/2)*sum(x_u,2)./(v(j).^2)); %
        pdf(:, j) = pdf(:, j) * w(j);        
    end
    % normalize pdf on x
    pSum_x = sum(pdf, 2);
    for j = 1:k
        pdf(:, j) = pdf(:, j)./pSum_x;
    end
    % normalize pdf on y
    pSum_y = sum(pdf, 1);
    pNorm = pdf * 0;
    for j = 1:k
        pNorm(:, j) = pdf(:, j)/pSum_y(j);
    end
    %存當下的值
    c_0=color;
    v_0=v;
    w_0=w;
    % update color
    color = (pNorm.') * data;
    %% M
    for j = 1:k
        for l = 1:dim
            x_u(:, l) = data(:, l) - color(j, l)*ones(n, 1);
        end
        x_u=x_u.*x_u;
        x_uSum=sum(x_u,2);
        v(j)=sqrt(1/dim*(pNorm(:,j).')*x_uSum); %
    end
    % update w
    w = (sum(pdf)/n).';
    
    iter = iter + 1;
    energy=sum(sum((color-c_0).^2))+sum(sum((v-v_0).^2))+(sum((w-w_0).^2));
end
end




%continue here


%%====================================================
