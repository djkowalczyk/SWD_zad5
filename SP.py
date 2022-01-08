import numpy as np
import SWD
class PointData:
    def __init__(self,dist=None,points=[],normalized_dist=None):
        self.dist = dist
        self.point = points
        self.normalized_dist = normalized_dist


def find_voronoi(A1,A2,points):
    def line(x,a,b):
        return a*x + b

    d = [(A2[0] - A1[0])/2,(A2[1] - A1[1])/2]
    d_sorted = sorted(d)
    u1 = A2[0] - d_sorted[0]
    l1 = A1[0] + d_sorted[0]
    u2 = A2[1] - d_sorted[0]
    l2 = A1[1] + d_sorted[0]
    C3 = np.linalg.solve(np.array([[A2[0],1],[u1,1]]),np.array([A2[1],u2]))
    C1 = np.linalg.solve(np.array([[A1[0],1],[l1,1]]),np.array([A1[1],l2]))
    if u1 == l1:
        line_constant_y = np.arange(l2,u2,0.01)
        line_constatnt = [(u1,i) for i in line_constant_y]
    elif u2 == l2:
        line_constant_x = np.arange(l1, u1, 0.01)
        line_constatnt = [(i, u2) for i in line_constant_x]
    line1_x = np.arange(A1[0], l1, 0.01)
    line2_x = np.arange(u1, A2[0], 0.01)
    line1 = [(i,line(i,C1[0],C1[1])) for i in line1_x]
    line2 = [(i, line(i, C3[0], C3[1])) for i in line2_x]
    func = line1 + line_constatnt + line2
    data = []
    for point in points:
        dist = np.inf
        if point[0] >= A1[0] and point[0] <= A2[0] and point[1] >= A1[1] and point[1] <= A2[1]:
            for x,y in func:
                dist_temp = ((point[0] - x)**2 + (point[1] - y)**2)**0.5
                if dist_temp < dist:
                    dist = dist_temp
                    actual_point = (x,y)

            normalized_value = (actual_point[1] - min(A1[1],A2[1]))/(max(A1[1],A2[1])-min(A1[1],A2[1]))
            data.append(PointData(dist,tuple(point),normalized_value))
    return data
def scoring(voronois):
    scor = {}
    for voronoi in voronois:
        for point_v in voronoi:
            if point_v.point in scor.keys():
                scor[point_v.point] = scor[point_v.point] + point_v.normalized_dist
            elif point_v.point not in scor.keys():
                scor[point_v.point] = point_v.normalized_dist
    return scor


def main():
    points = [(0, 2), (1, 2), (1, 5), (2, 3), (2, 9), (3, 1), (3, 6), (3, 8), (4, 3), (4, 5), (3, 6), (5, 7), (6, 9), (6, 10), (5, 3), (7, 5), (7, 10), (8, 8), (9, 2), (9, 5), (9, 7), (9, 9), (10, 4), (10, 8), (10, 9), (11, 6), (11, 10), (12, 1), (12, 4), (12, 7)]
    limit = (len(points)//4) + 1
    SWD.sorting_points(points)
    A1, A2, U = SWD.divide_into_groups(points, limit)
    # areas = SWD.field_of_square(A1, A2, U)
    # print(areas)
    # areas = SWD.standardization_of_squares(areas)
    # # areas = ranking_creating(areas)
    print(A1)
    print(A2)
    print(U)
    data = []
    for a in A1:
        for b in A2:
            data.append(find_voronoi(a, b, U))
    sc = scoring(data)
    print(sc)


if __name__ == "__main__":

    main()