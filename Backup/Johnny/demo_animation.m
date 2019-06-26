pos=randn(3,30000,101)*40;



figure('position',[300 300 512 512])
scatter3(pos(1,:,1),pos(2,:,1),pos(3,:,1))
xlabel('x pos')
ylabel('y pos')
zlabel('z pos')
F(1)=getframe()

for i=2:101;
    tic
    scatter3(pos(1,:,i),pos(2,:,i),pos(3,:,i))
    xlabel('x pos')
    ylabel('y pos')
    zlabel('z pos')
    drawnow
    F(i)=getframe(gcf)
    %pause(0.1)
    toc 
end

video=VideoWriter('demo','Archival')
open(video)
writeVideo(video,F)
close(video)