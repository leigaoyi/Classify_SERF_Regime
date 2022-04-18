clear all;
load data/serf/20220301_150~141_0.2_00000.mat;

grid_all = dev4835.demods(1).sample{1, 1}.grid*9.21;
first_x = [];
for j = 1:1
     for i = 1:152  % 152 serf  ; 76 non serf 150, 124 140du
     second_x = dev4835.demods(j).sample{1, i}.x;%第一次扫场数据有问题
     %plot(grid_all,x_all);
     %hold on;
     first_x = cat(1, first_x, second_x);
     end
end


xlswrite('data/serf/serf_152.xlsx',first_x,'sheet1','A1');
