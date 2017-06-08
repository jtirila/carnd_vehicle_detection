def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Cycles through an image using the scheme determined by the parameters, and returns a list of 
    bounding boxes This is useful for systematic scanning of subregions of an image.
    
    :param x_start_stop: a two element container specifying the boundaries of the region to traverse in x-direction. 
           Both can be None, in this case the edges of the image are used.
    :param y_start_stop: see x_start_stop, just now this is for the y-direction
    
    :return: A list of ((top-left-x, top-left-y), (bottom-right-x, bottom-right-y)) coordinate tuples.
    """
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop[0] = 0 if x_start_stop[0] is None else x_start_stop[0]
    x_start_stop[1] = img.shape[1] if x_start_stop[1] is None else x_start_stop[1]
    y_start_stop[0] = 0 if y_start_stop[0] is None else y_start_stop[0]
    y_start_stop[1] = img.shape[0] if y_start_stop[1] is None else y_start_stop[1]

    step_x, step_y = tuple(np.product(x) for x in zip(xy_window, xy_overlap))
    num_win_x = (x_start_stop[1] - x_start_stop[0] - xy_window[0]) // step_x + 1
    num_win_y = (y_start_stop[1] - y_start_stop[0] - xy_window[1]) // step_y + 1

    print("Num win x: {}".format(num_win_x))
    print("Num win y: {}".format(num_win_y))

    y_win_top = y_start_stop[0]
    y_win_bottom = y_win_top + xy_window[1]

    window_list = []
    while True:
        x_win_left = x_start_stop[0]
        x_win_right = x_win_left + xy_window[0]
        if y_win_bottom - 1 > y_start_stop[1]:
            break
        while True:
            if x_win_right - 1 > x_start_stop[1]:
                break

            window_list.append(((x_win_left, y_win_top), (x_win_right, y_win_bottom)))

            # Increment the left and right positions of the window by step size in x direction
            x_win_left, x_win_right = tuple([int(sum(x)) for x in zip((x_win_left, x_win_right), (step_x, ) * 2)])

        # Increment the top and bottom positions of the window by step size in y direction
        y_win_top, y_win_bottom = tuple([int(sum(x)) for x in zip((y_win_top, y_win_bottom), (step_y, ) * 2)])
    return window_list
