from utils import *

class Curve():
    def __init__(self, curve, t_range=[0, 1], t_step=0.01):
        self.Y = curve
        self.t_range = t_range
        self.t_step = t_step
        self.T = self.tangent_vector()
        self.N = self.normal_vector()
        self.B = self.binormal_vector()
        self.k = self.curvature()
        self.Tau = self.torsion()

    def evaluate(self, x_expr, y_expr, z_expr, t):
        x = sp.sympify(x_expr)
        y = sp.sympify(y_expr)
        z = sp.sympify(z_expr)
        x_val = x.subs('t', t)
        y_val = y.subs('t', t)
        z_val = z.subs('t', t)
        return np.array([x_val, y_val, z_val])

    def parametrized_curve(self, Y=None, t_range=[0, 1], t_step=0.01):
        if Y is None:
            Y = self.Y
        t = np.arange(t_range[0], t_range[1], t_step)
        Y_points = np.array([self.evaluate(Y[0], Y[1], Y[2], t_i) for t_i in t])
        return Y_points

    def tangent_vector(self, Y=None):
        if Y is None:
            Y = self.Y
        Y_ = diff(Y, 't')
        T = normalize(Y_)
        return T

    def normal_vector(self, Y=None):
        if Y is None:
            Y = self.Y
        T = self.tangent_vector(Y)
        T_ = diff(T, 't')
        N = normalize(T_)
        return N

    def binormal_vector(self, Y=None):
        if Y is None:
            Y = self.Y
        T = self.tangent_vector(Y)
        N = self.normal_vector(Y)
        B = sp.simplify(np.cross(T, N))
        return B

    def curvature(self, Y=None):
        if Y is None:
            Y = self.Y
        Y_ = diff(Y, 't')
        Y__ = diff(Y_, 't')
        k = sp.simplify(norm(np.cross(Y__, Y_)) / norm(Y_) ** 3)
        return k

    def torsion(self, Y=None):
        if Y is None:
            Y = self.Y
        Y_ = diff(Y, 't')
        Y__ = diff(Y_, 't')
        Y___ = diff(Y__, 't')
        Tau = sp.simplify(np.dot(np.cross(Y_, Y__), Y___) / norm(np.cross(Y_, Y__)) ** 2)
        return Tau

    def plot_curve(self, Y=None, t_range=[0, 1], t_step=0.01, p=None, draw_tangent=False, draw_normal=False,
                   draw_binormal=False, verbose=False):
        if Y is None:
            Y = self.Y
        # Basics
        Y_points = self.parametrized_curve(Y, t_range, t_step)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot(Y_points[:, 0], Y_points[:, 1], Y_points[:, 2])

        # Get point on curve
        point = self.evaluate(Y[0], Y[1], Y[2], p if p is not None else t_range[0])

        # Draw tangent vector
        if draw_tangent:
            tangent = self.tangent_vector(Y)
            tangent_direction = self.evaluate(tangent[0], tangent[1], tangent[2], p if p is not None else t_range[0])
            ax.quiver(point[0], point[1], point[2], tangent_direction[0], tangent_direction[1],
                      tangent_direction[2], length=1, normalize=True, color='red')
            if verbose:
                print(f"Tangent vector: {tangent}")
                print(f"Tangent direction: {tangent_direction}")

        # Draw normal vector
        if draw_normal:
            normal = self.normal_vector(Y)
            normal_direction = self.evaluate(normal[0], normal[1], normal[2], p if p is not None else t_range[0])
            ax.quiver(point[0], point[1], point[2], normal_direction[0], normal_direction[1],
                      normal_direction[2], length=1, normalize=True, color='orange')
            if verbose:
                print(f"Normal vector: {normal}")
                print(f"Normal direction: {normal_direction}")

        # Draw binormal vector
        if draw_binormal:
            binormal = self.binormal_vector(Y)
            binormal_direction = self.evaluate(binormal[0], binormal[1], binormal[2], p if p is not None else t_range[0])
            ax.quiver(point[0], point[1], point[2], binormal_direction[0], binormal_direction[1],
                      binormal_direction[2], length=1, normalize=True, color='gold')
            if verbose:
                print(f"Binormal vector: {binormal}")
                print(f"Binormal direction: {binormal_direction}")

        # Draw point on curve
        ax.scatter(point[0], point[1], point[2], color='black')

        # Write axes names
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Make all axes equal
        max_range = float(np.array([Y_points[:, 0].max() - Y_points[:, 0].min(),
                                    Y_points[:, 1].max() - Y_points[:, 1].min(),
                                    Y_points[:, 2].max() - Y_points[:, 2].min()]).max() / 2.0)
        mid_x = float((Y_points[:, 0].max() + Y_points[:, 0].min()) * 0.5)
        mid_y = float((Y_points[:, 1].max() + Y_points[:, 1].min()) * 0.5)
        mid_z = float((Y_points[:, 2].max() + Y_points[:, 2].min()) * 0.5)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()


if __name__ == "__main__":
    circular_helix = Curve(['cos(t)', 'sin(t)', 't'])
    circular_helix.plot_curve(t_range=[0, 2 * np.pi], t_step=0.1, p=2.1 * np.pi / 2, draw_tangent=True,
                              draw_normal=True, draw_binormal=True)
