function plot_heat(matrix,heading,rows,columns)
    figure;
    imagesc(matrix);
    title(heading);
    for i = 1:length(rows)
        ylabels{i} = rows(i);
    end
    for j = 1:length(columns)
        xlabels{j} = columns(j);
    end
    
    textStrings = num2str(matrix(:),'%0.2f');  %# Create strings from the matrix values
    textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
    
    [x,y] = meshgrid(1:length(columns),1:length(rows));   %# Create x and y coordinates for the strings
    hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                    'HorizontalAlignment','center');
    midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
    textColors = repmat(matrix(:) < midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
    set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

    set(gca,'xtick',1:1:length(columns))
    set(gca,'ytick',1:1:length(rows))
    set(gca, 'xTickLabel', columns);
    xlabel('Learning Rate');
    set(gca, 'yTickLabel', rows);
    ylabel('Batch Size');
    colorbar;
end
