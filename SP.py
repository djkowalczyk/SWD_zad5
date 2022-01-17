import numpy as np
class PointData:
    def __init__(self,dist=None,points=[],normalized_dist=None):
        self.dist = dist
        self.point = points
        self.normalized_dist = normalized_dist

def sorting_points(points):
    return sorted(points, key=lambda x: (x[0] ** 2 + x[1] ** 2) ** 0.5)

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
    if round(u1,5) == round(l1,5):
        if u2 < l2:
            u2,l2 = l2,u2
        line_constant_y = np.arange(l2,u2,0.01)
        line_constatnt = [(u1,i) for i in line_constant_y]
    elif round(u2,5) == round(l2,5):
        if u1 < l1:
            u1,l1 = l1,u1
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
        # if point[0] >= A1[0] and point[0] <= A2[0] and point[1] >= A1[1] and point[1] <= A2[1]:
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

def divide_into_groups(points, limit: int):
    considered_points = points[limit:-limit]
    #ograniczenie zbiorów a następnie podzielenie ich
    candidates_a1 = considered_points[:len(considered_points)//2]
    candidates_a2 = considered_points[len(considered_points)//2:]

    out1 = check_if_points_independant(candidates_a1)
    #usuniecie punktow zaleznych z a1
    a1 = [i for i in candidates_a1 if i not in out1]
    out2 = check_if_points_independant(candidates_a2)
    u = out1
    #usuniecie punktow zaleznych z a2
    a2 = [i for i in candidates_a2 if i not in out2]
    return a1, a2, u


def check_if_points_independant(points):
    # funkcja sprawdzająca  kolejny punkt po punkcie czy kazdy z nich jets punktem nie zaleznym w danym podzbiorze
    index = []
    for i in points:
        for j in points:
            if i[0] <= j[0] and i[1] <= j[1] and j not in index and i != j:
                index.append(j)
            elif i[0] >= j[0] and i[1] >= j[1] and i not in index and i != j:
                index.append(i)
    return index

def main(points):
    limit = (len(points)//4) + 1
    sorting_points(points)
    A1, A2, U = divide_into_groups(points, limit)
    data = []
    for a in A1:
        for b in A2:
            data.append(find_voronoi(a, b, U))
    sc = scoring(data)
    return sc,A1,A2


if __name__ == "__main__":

    main()