function [ val ] = weightedAverage(weights, values)
%WEIGHTEDAVERAGE Calculate the weighted average of 'values' by applying 
% the 'weights'
%
%   values - Data points to average, one per row.
%  weights - Weight to apply to each data point, one per row.
%
%  Returns:
%     val  - The weighted average of 'values'.

    % Apply the weights to the values by taking the dot-product between the
    % two vectors.
    val = weights' * values; % 權重乘上值

    % Divide by the sum of the weights.
    val = val ./ sum(weights, 1); % 把已經乘上權重的值除以所有的權重來或的 wieghted average
    
end

