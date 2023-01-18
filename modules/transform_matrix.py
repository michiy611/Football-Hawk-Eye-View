


def transform_matrix(matrix, p, vid_shape, gt_shape):
  p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
  px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
  py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

  p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))

  return p_after