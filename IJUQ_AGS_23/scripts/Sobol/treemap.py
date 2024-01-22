import openturns as ot
import numpy as np


class RectangleInUnitSquare:
    def __init__(self, origin_x, origin_y, width, height):
        """
        Create a rectangle in the unit square [0, 1] x [0, 1]

        Parameters
        ----------
        origin_x : float, in [0, 1]
            The X coordinate of the origin of the rectangle.
        origin_y : float, in [0, 1]
            The Y coordinate of the origin of the rectangle.
        width : float, in [0, 1]
            The width of the rectangle i.e. the horizontal length
            on the X axis.
        height : float, in [0, 1]
            The height of the rectangle i.e. the vertical length
            on the Y axis.
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        if origin_x < 0.0:
            raise ValueError(f"The X origin is {origin_x} < 0.")
        if origin_x > 1.0:
            raise ValueError(f"The X origin is {origin_x} > 1.")
        if origin_y < 0.0:
            raise ValueError(f"The Y origin is {origin_y} < 0.")
        if origin_y > 1.0:
            raise ValueError(f"The Y origin is {origin_y} > 1.")
        if width < 0.0:
            raise ValueError(f"The width is {width} < 0.")
        if width > 1.0 + ot.SpecFunc.ScalarEpsilon:
            raise ValueError(f"The width is {width} > 1.")
        if height < 0.0:
            raise ValueError(f"The height is {height} < 0.")
        if height > 1.0 + ot.SpecFunc.ScalarEpsilon:
            raise ValueError(f"The height is {height} > 1.")
        return

    def __repr__(self):
        s = "RectangleInUnitSquare("
        s += f"x : {self.origin_x:.4f}, "
        s += f"y : {self.origin_y:.4f}, "
        s += f"w : {self.width:.4f}, "
        s += f"h : {self.height:.4f}"
        s += ")"
        return s


class TreeMap:
    def Rectangle2Polygon(rectangle):
        """
        Create a Polygon from the parmeters of a rectangle.

        Parameters
        ----------
        rectangle : RectangleInUnitSquare
            A rectangle in the unit square [0, 1] x [0, 1]

        Returns
        -------
        polygon : ot.Polygon
            The rectangle.
        """
        data_square = [
            [rectangle.origin_x, rectangle.origin_y],
            [rectangle.origin_x + rectangle.width, rectangle.origin_y],
            [
                rectangle.origin_x + rectangle.width,
                rectangle.origin_y + rectangle.height,
            ],
            [rectangle.origin_x, rectangle.origin_y + rectangle.height],
        ]
        polygon = ot.Polygon(data_square)
        return polygon

    def __init__(
        self,
        valueList,
        labelList,
        sum_threshold=1.0e-8,
        text_size=1.0,
        line_width=2.0,
        verbose=False,
        fill_color="white",
        edge_color="black",
        text_color="black",
        fillWithColor=False,
    ):
        """
        Create a TreeMap of proportion valueList.

        Each value is in [0, 1] and the sum is equal to one,
        up to the tolerance.

        Parameters
        ----------
        valueList : sequence of floats, with length n
            The nonnegative valueList which sum to 1.
        group_labels : list(str), length n
            For each value, its string label. The default is None.
        text_size : float > 0.0, optional
            The size of the text. The default is 1.0.
        line_width : float > 0.0, optional
            The line width. The default is 2.0.
        verbose : bool, optional
            If True, print intermediate messages. The default is False.
        sum_threshold : float > 0.0, optional
            The tolerance for the sum of the proportions.
            The sum of proportions is expected to be close to 1, with absolute tolerance equal
            to sum_threshold.
            Default is 1.e-8.
        fill_color : str
            The default color to fill each rectangle.
        edge_color : str
            The default color of the edges.
        text_color : str
            The default color of the text.
        fillWithColor : bool
            Set to True to fill the rectangles with colors.

        Returns
        -------
        graph : ot.Graph()
            A sensitivity Treemap.

        """
        self.text_size = text_size
        self.line_width = line_width
        self.verbose = verbose
        self.fill_color = fill_color
        self.edge_color = edge_color
        self.text_color = text_color
        self.fillWithColor = fillWithColor
        if sum(valueList) > 1.0 + sum_threshold:
            raise ValueError(
                f"Warning : The sum of valueList is {sum(valueList)} > 1.0",
            )
        if sum(valueList) >= 1.0:
            if verbose:
                print("Set the last so that the sum is exactly 1.")
            valueList[-1] = 1.0 - sum(valueList[0:-1])
        if len(valueList) != len(labelList):
            raise ValueError(
                "Number of valueList is %d" % (len(valueList)),
                "but number of labelList is %d" % (len(labelList)),
            )
        if abs(sum(valueList) - 1.0) > sum_threshold:
            raise ValueError(
                "The sum of proportions is %f, which is different from 1."
                % (sum(valueList))
            )
        # Sort in decreasing order according to valueList
        values_sample = ot.Sample.BuildFromPoint(valueList)
        isIncreasing = False
        indices = values_sample.argsort(isIncreasing)
        self.valueList = [values_sample[i, 0] for i in indices]
        self.labelList = [labelList[i] for i in indices]
        return

    def columnwise(self, items_by_column):
        """
        Create the columnwise TreeMap of valueList

        Parameters
        ----------
        items_by_column : list(int)
            The number of items to plot on each line or column.

        Returns
        -------
        graph : ot.Graph()
            A Treemap.

        """
        if len(self.valueList) != sum(items_by_column):
            raise ValueError(
                "Number of valueList is %d" % (len(self.valueList)),
                "but number of items is %d" % (sum(items_by_column)),
            )
        for i in range(len(items_by_column)):
            if items_by_column[i] < 1:
                raise ValueError(
                    "Number of valueList in line %i is %d"
                    " but the minimum is 1." % (i, items_by_column[i])
                )

        # Compute the rectangles
        rectangle_list = self.computeColumnwise(items_by_column)
        # Create the list of texts
        text_list = []
        for i in range(len(rectangle_list)):
            rectangle_i = rectangle_list[i]
            text_i = ot.Text(
                [rectangle_i.origin_x + rectangle_i.width / 2],
                [rectangle_i.origin_y + rectangle_i.height / 2],
                [self.labelList[i]],
            )
            text_list.append(text_i)
        # Create the TreeMap
        graph = self.createTreeMapFromRectangles(rectangle_list, text_list)
        return graph

    def computeColumnwise(self, items_by_column):
        """
        Compute the columnwise TreeMap of valueList

        Parameters
        ----------
        items_by_column : list(int)
            The number of items to plot on each line or column.
            The sum of elements in the list must be equal to the number
            of valueList to represent.

        Returns
        -------
        rectangle_list : list(RectangleInUnitSquare)
            The list of rectangles within the TreeMap.

        text_list : list(ot.Text)
            The list of texts.
        """
        if sum(items_by_column) != len(self.valueList):
            raise ValueError(
                f"Number of valueList to represent is {len(self.valueList)} "
                f"but the number of elements to be plotted is {sum(items_by_column)}"
            )
        # Initialize
        rectangle_list = []
        """
        h
        e  +---------+
        i  |         |
        g  |         |
        h  |         |
        t  +---------+
            width
        origin
        """
        origin = ot.Point([0.0] * 2)
        width = 1.0
        height = 1.0
        index = 0
        n_lines = len(items_by_column)
        for i in range(n_lines):
            n_items = items_by_column[i]
            area = sum(self.valueList[index : index + n_items])
            width = area / height
            if self.verbose:
                print("+ i = ", i)
                print("  Number of items = ", n_items)
                print("  area = ", area, ", width = ", width)
            # Plot sensitivity indices in the column
            cumulated_height = origin[1]
            for j in range(n_items):
                item_height = self.valueList[index + j] / width
                if self.verbose:
                    print("  + j = ", j)
                    print("    item_height = ", item_height)
                rectangle = RectangleInUnitSquare(
                    origin[0], cumulated_height, width, item_height
                )
                rectangle_list.append(rectangle)
                # Update to next sensitivity index in the same column
                cumulated_height += item_height
            # Update to next column
            index += n_items
            origin[0] += width
        return rectangle_list

    def computeAlternateDirections(self):
        """
                Compute the alternating directions TreeMap of valueList
        v
                Returns
                -------
                rectangle_list : list(RectangleInUnitSquare)
                    The list of rectangles within the TreeMap.

                text_list : list(ot.Text)
                    The list of texts.

        """
        rectangle_list = []
        text_list = []

        # Initialize
        """
        h
        e  +---------+
        i  |         |
        g  |         |
        h  |         |
        t  +---------+
              width
        origin
        """
        origin = ot.Point([0.0] * 2)  # bottom left origin
        width = 1.0
        height = 1.0
        odd = True  # If odd, divide horizontally, otherwise vertically
        # Loop over the indices
        for i in range(len(self.valueList)):
            value = self.valueList[i]
            if self.verbose:
                print(
                    "+ i = %d" % (i),
                    ", origin %s" % (origin),
                    ", width = %.4f" % (width),
                    ", height = %.4f" % (height),
                    ", value = %.4f" % (value),
                    " : ",
                    self.labelList[i],
                )
            if odd:
                """
                Divide horizontally
                +----+----+
                |    |    |
                |    |    |
                |    |    |
                +----+----+
                """
                odd = False
                division_width = value / height
                rectangle = RectangleInUnitSquare(
                    origin[0], origin[1], division_width, height
                )
                rectangle_list.append(rectangle)
                text = ot.Text(
                    [origin[0] + division_width / 2.0],
                    [origin[1] + height / 2],
                    [self.labelList[i]],
                )
                text_list.append(text)
                origin[0] += division_width
                width -= division_width
                if self.verbose:
                    print("    odd is False : divide horizontally")
                    print("    Move the origin right +%.4f" % (division_width))
                    area = division_width * height
                    print(
                        "    Area = %.4f" % (division_width),
                        " x %.4f" % (height),
                        " = %4f" % (area),
                    )
            else:
                odd = True
                """
                Divide vertically
                +---------+
                |         |
                +---------+
                |         |
                +---------+
                """
                division_height = value / width
                rectangle = RectangleInUnitSquare(
                    origin[0], origin[1], width, division_height
                )
                rectangle_list.append(rectangle)
                text = ot.Text(
                    [origin[0] + width / 2.0],
                    [origin[1] + division_height / 2],
                    [self.labelList[i]],
                )
                text_list.append(text)
                origin[1] += division_height
                height -= division_height
                if self.verbose:
                    print("    odd is True : divide vertically")
                    print("    Move the origin up +%.4f" % (height))
                    area = width * division_height
                    print(
                        "    Area = %.4f" % (width),
                        " x %.4f" % (division_height),
                        " = %4f" % (area),
                    )
        return rectangle_list, text_list

    def alternateDirections(self):
        """
        Create the alternating directions TreeMap of valueList

        Returns
        -------
        graph : ot.Graph()
            A Treemap.

        """
        rectangle_list, text_list = self.computeAlternateDirections()
        graph = self.createTreeMapFromRectangles(rectangle_list, text_list)
        return graph

    def createTreeMapFromRectangles(self, rectangle_list, text_list):
        """
        Create a TreeMap from a list of rectangles and texts

        Parameters
        ----------
        rectangle_list : list(RectangleInUnitSquare)
            The list of rectangles within the TreeMap.

        text_list : list(ot.Text)
            The list of texts.

        Returns
        -------
        graph : ot.Graph()
            A Treemap.
        """
        # Plot
        number_of_rectangles = len(rectangle_list)
        palette = ot.Drawable.BuildDefaultPalette(number_of_rectangles)
        graph = ot.Graph("TreeMap", "", "", True)
        for i in range(number_of_rectangles):
            polygon = TreeMap.Rectangle2Polygon(rectangle_list[i])
            if self.fillWithColor:
                polygon.setColor(palette[i])
            else:
                polygon.setColor(self.fill_color)
            polygon.setEdgeColor(self.edge_color)
            graph.add(polygon)
            if self.text_size > 0.0:
                text_i = text_list[i]
                text_i.setTextSize(self.text_size)
                text_i.setColor(self.text_color)
                graph.add(text_i)
        graph.setGrid(False)
        graph.setAxes(False)
        return graph

    def computeShapeScore(self, items_by_column, score="mean", verbose=False):
        """
        Compute the shape ratio of the TreeMap

        For a given rectangle, the shape is a number greater than 1
        defined by the equation:

        height / width,    if height > width,
        width / height,    otherwise.

        The score of a treemap depends on the value of the shape of
        all its rectangles.

        Parameters
        ----------
        items_by_column : list(int)
            The number of items in each column.
            The sum of its elements must be equal to the number of valueList.
        score : str
            The max score is the maximum of the ratios of the shapes of the rectangles.
            The mean score is the mean of the ratios of the shapes of the rectangles.
        verbose : bool
            Set to True to print intermediate messages.

        """
        if sum(items_by_column) != len(self.valueList):
            raise ValueError(
                f"Number of valueList to represent is {len(self.valueList)} "
                f"but the number of elements to be plotted is {sum(items_by_column)}"
            )
        # Remove zero valueList
        non_zero_items_by_line = []
        for nb_items in items_by_column:
            if nb_items > 0:
                non_zero_items_by_line.append(nb_items)
        # Call the TreeMap
        if verbose:
            print("non_zero_items_by_line = ", non_zero_items_by_line)
            print("Number of actual columns = ", len(non_zero_items_by_line))
        rectangle_list = self.computeColumnwise(non_zero_items_by_line)
        # Compute shape ratio
        shape_ratios = []
        for i in range(len(rectangle_list)):
            r = rectangle_list[i]
            if r.width > r.height:
                ratio = r.width / r.height
            else:
                ratio = r.height / r.width
            if verbose:
                print(f"i = {i}, ratio = {ratio:.2f}")
            shape_ratios.append(ratio)
        if score == "max":
            worst_shape_ratio = max(shape_ratios)
        elif score == "mean":
            worst_shape_ratio = sum(shape_ratios) / len(shape_ratios)
        else:
            raise ValueError(f"Unknown score = {score}.")
        return worst_shape_ratio

    def searchForBestNumberOfItemsByLine(
        self,
        maximumNumberOfColumns,
        score="mean",
        algorithm="auto",
        maximumNumberOfRandomTrials=20,
        verbose=False,
        default_maximum_best_sample=24,
        default_maximum_random_sample_size=35,
    ):
        """
        Search for the best number of items by column

        We search for the multi-index items_by_column of dimension maximumNumberOfColumns
        such that:

        sum(items_by_column) = numberOfValues

        where numberOfValues is the number of valueList to represent in the
        TreeMap.
        We use the LinearEnumerateFunction class to provide the
        collection of multi-indices having this property.

        If we directly use LinearEnumerateFunction, some indices of
         the multi-index are equal to 0, for example (3, 0, 4).
        This is uninteresting in our case, because this means that some
        column does not contain any particular value.
        In the example where the multi-index is equal to (3, 0, 4),
        this means that the first column has 3 valueList and the third has 4 valueList,
        but the second column is empty.
        So we arrange the algorithm so that the minimum marginal index
        is 1, by adding +1 to each component.
        This increases the sum by the dimension of the multi-index,
        equal to maximumNumberOfColumns.
        This is why we consider a shifted strata index.

        Example
        -------
        >>> valueList = [0.15] * 2 + [0.12] * 3 + [0.07] * 2 + [0.005] * 5
        >>> sum_values = sum(valueList)
        >>> valueList = [v / sum_values for v in valueList]  # Standardize
        >>> groups_list = list(range(len(valueList)))
        >>> labelList = [str(group) for group in groups_list]
        >>> treemap = tm.TreeMap(valueList, labelList)
        >>> items_by_line = treemap.searchForBestNumberOfItemsByLine(
                maximumNumberOfColumns, verbose=True
            )
        Start = 165, stop = 220, number = 55
        i = 165, items_by_column = [10,1,1], cost = 165.0000
        + Best !
        i = 166, items_by_column = [9,2,1], cost = 165.0000
        i = 167, items_by_column = [9,1,2], cost = 165.0000
        + Best !
        i = 168, items_by_column = [8,3,1], cost = 165.0000
        i = 169, items_by_column = [8,2,2], cost = 157.0970
        + Best !
        i = 170, items_by_column = [8,1,3], cost = 165.0000
        i = 171, items_by_column = [7,4,1], cost = 165.0000
        i = 172, items_by_column = [7,3,2], cost = 41.2500
        + Best !
        i = 173, items_by_column = [7,2,3], cost = 41.2500
        [...]
        i = 191, items_by_column = [4,2,6], cost = 6.3989
        + Best !
        i = 192, items_by_column = [4,1,7], cost = 6.8750
        [...]
        i = 219, items_by_column = [1,1,10], cost = 66.8182
        Best combination :  [4,2,6]

        Parameters
        ----------
        maximumNumberOfColumns : int
            The maximum number of columns to consider.
        verbose : bool
            Set to True to print intermediate messages.

        Returns
        -------
        items_by_column : list(int)
            The number of items to plot on each line or column.
            The sum of elements in the list must be equal to the number
            of valueList to represent.

        """
        # Initialize
        bestCost = np.inf
        best_items_by_column = None
        numberOfValues = len(self.valueList)
        # Special cases
        if numberOfValues == 1:
            best_items_by_column = [1]
            return best_items_by_column
        if numberOfValues <= maximumNumberOfColumns:
            best_items_by_column = [1] * numberOfValues
            return best_items_by_column
        if algorithm == "auto":
            # Switch to the 3 strategies depending on the sample size
            if numberOfValues < default_maximum_best_sample:
                algorithm = "best"
            elif numberOfValues < default_maximum_random_sample_size:
                algorithm = "random"
            else:
                algorithm = "fast"
        if algorithm == "fast":
            number_of_columns = int(np.sqrt(numberOfValues))
            best_items_by_column = [number_of_columns] * number_of_columns
            best_items_by_column[-1] = numberOfValues - number_of_columns * (
                number_of_columns - 1
            )
        else:
            # Prepare enumerate function
            enumerateFunction = ot.LinearEnumerateFunction(maximumNumberOfColumns)
            starting_index = enumerateFunction.getStrataCumulatedCardinal(
                numberOfValues - 1 - maximumNumberOfColumns
            )
            stopping_index = enumerateFunction.getStrataCumulatedCardinal(
                numberOfValues - maximumNumberOfColumns
            )
            number_of_indices = stopping_index - starting_index
            if verbose:
                print(
                    f"Start = {starting_index}, stop = {stopping_index}, number = {number_of_indices}"
                )
            if algorithm == "best":
                # Loop over all combinations
                indices_collection = list(range(starting_index, stopping_index))
            elif algorithm == "random":
                # Generate a uniform discrete sample over the candidate indices
                distribution = ot.UserDefined(
                    [[i] for i in range(starting_index, stopping_index)]
                )
                indices_sample = distribution.getSample(maximumNumberOfRandomTrials)
                indices_collection = [
                    int(indices_sample[i, 0])
                    for i in range(maximumNumberOfRandomTrials)
                ]
            else:
                raise ValueError(f"Unknown algorithm {algorithm}.")
            # For each items_by_column, we have sum(items_by_column) = numberOfValues
            for i in indices_collection:
                items_by_column = enumerateFunction(i)
                items_by_column = ot.Indices([v + 1 for v in items_by_column])
                cost = self.computeShapeScore(items_by_column, score)
                if verbose:
                    print(
                        f"i = {i}, items_by_column = {items_by_column}, cost = {cost:.4f}"
                    )
                if cost < bestCost:
                    if verbose:
                        print("+ Best !")
                    bestCost = cost
                    best_items_by_column = items_by_column
        return best_items_by_column
