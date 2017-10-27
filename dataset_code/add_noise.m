function [X_dash,y_dash] = add_noise(X,y,n_dim,noise_mean,noise_std)
% This function adds noise to the dataset in terms of noise columns.
% n_dim columns are sampled from a Gaussian distribution with noise_mean
% and noise_std
    
    X_dash = X;
    y_dash = y;
    for i=1:n_dim
        X_dash(:,i+size(X,2))=normrnd(noise_mean,noise_std,size(X_dash,1),1);
    end
end

