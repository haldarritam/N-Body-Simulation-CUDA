num_bodies=100;
time_step=100;
figure('position',[300 300 512 512]);
fp=fopen('pos.txt','r');
%file format based on stephten's code
%x y z coordinate each line for each bodies 


for i=1:time_step
    for j=1:num_bodies
        pos(:,j,i)=str2num(fgetl(fp));
    end
end

fclose(fp);

for i=1:time_step
    tic
    scatter3(pos(1,:,i),pos(2,:,i),pos(3,:,i))
    %scatter3(pos(2,:,i),pos(3,:,i),pos(4,:,i))
    xlabel('x pos')
    ylabel('y pos')
    zlabel('z pos')
    xlim([-1000 1000])
    ylim([-1000 1000])
    zlim([-1000 1000])
    drawnow
    F(i)=getframe(gcf)
    %pause(0.1)
    toc 
end

video=VideoWriter('demo','MPEG-4');
%we can use fps to show the speed difference between cpu and gpu
video.FrameRate=3.5;
open(video)
writeVideo(video,F)
close(video)


