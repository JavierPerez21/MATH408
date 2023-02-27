from utils import *

class Curve():
    def __init__(self, curve, t_range=[0, 1], t_steps=0.01):
        print("Initializing curve...", curve)
        assert self.is_regular(curve)[0], f"Curve is not regular! {self.is_regular(curve)[1]}"
        self.Y = self.make_unit_speed(curve)
        self.t_range = t_range
        self.t_steps = t_steps
        self.T = self.tangent_vector()
        self.N = self.normal_vector()
        self.B = self.binormal_vector()
        self.k = self.curvature()
        self.Tau = self.torsion()
        print("Curve initialized.")
        print("t in", self.t_range, "with step", self.t_steps)
        print("Y: ", self.Y)
        print("T: ", self.T)
        print("N: ", self.N)
        print("B: ", self.B)
        print("k: ", self.k)
        print("Tau: ", self.Tau)


    def make_unit_speed(self, Y=None):
        if Y is None:
            Y = self.Y
        Y_ = diff(Y, 't')
        if norm(Y_) == 1:
            return sp.sympify(Y)
        else:
            Y_norm = norm(Y_)
            int = sp.integrate(Y_norm, sp.Symbol('t'))
            s = int - int.subs('t', 0)
            t_s = sp.simplify(sp.solve(s - sp.Symbol('s'), sp.Symbol('t'))[-1])
            Y = [sp.simplify(y.replace('t', str(t_s))) for y in Y]
            Y = [sp.simplify(y.replace(sp.Symbol('s'), sp.Symbol('t'))) for y in Y]
            return Y

    def is_regular(self, Y=None):
        if Y is None:
            Y = self.Y
        Y_ = diff(Y)
        sols = [sp.solve(y, sp.Symbol('t')) for y in Y_]
        sol = set(sols[0]).intersection(*sols[1:])
        if len(sol) == 0:
            return True, None
        else:
            return False, sol

    def evaluate(self, x_expr, y_expr, z_expr, t):
        x = sp.sympify(x_expr)
        y = sp.sympify(y_expr)
        z = sp.sympify(z_expr)
        x_val = x.subs('t', t)
        y_val = y.subs('t', t)
        z_val = z.subs('t', t)
        return np.array([x_val, y_val, z_val])

    def parametrized_curve(self, Y=None, t_range=[0, 1], t_steps=0.01):
        if Y is None:
            Y = self.Y
        t = np.arange(t_range[0], t_range[1], t_steps)
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

    def get_osculating_circle(self, t0):
        R = self.k.subs('t', t0)
        Ys0 = self.evaluate(self.Y[0], self.Y[1], self.Y[2], t0)
        v1 = self.evaluate(self.T[0], self.T[1], self.T[2], t0)
        v2 = self.evaluate(self.N[0], self.N[1], self.N[2], t0)
        p = Ys0 + R*v2
        oc = p + R * v1 * sp.cos((sp.Symbol('t')-t0)/R) + R * v2 * sp.sin((sp.Symbol('t')-t0)/R)
        return oc

    def plot_curve(self, Y=None, t_range=None, t_steps=None, p=None, draw_tangent=False, draw_normal=False,
                   draw_binormal=False, verbose=False, osculating_circle=True):
        if Y is None:
            Y = self.Y
        if t_range is None:
            t_range = self.t_range
        if t_steps is None:
            t_steps = self.t_steps
        print("Evaluating at", p)
        # Basics
        Y_points = self.parametrized_curve(Y, t_range, t_steps)
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
                      tangent_direction[2], length=1, normalize=True, color='red', label='T')
            if verbose:
                print(f"Tangent vector: {tangent}")
                print(f"Tangent direction: {tangent_direction}")

        # Draw normal vector
        if draw_normal:
            normal = self.normal_vector(Y)
            normal_direction = self.evaluate(normal[0], normal[1], normal[2], p if p is not None else t_range[0])
            ax.quiver(point[0], point[1], point[2], normal_direction[0], normal_direction[1],
                      normal_direction[2], length=1, normalize=True, color='orange', label='N')
            if verbose:
                print(f"Normal vector: {normal}")
                print(f"Normal direction: {normal_direction}")

        # Draw binormal vector
        if draw_binormal:
            binormal = self.binormal_vector(Y)
            binormal_direction = self.evaluate(binormal[0], binormal[1], binormal[2], p if p is not None else t_range[0])
            ax.quiver(point[0], point[1], point[2], binormal_direction[0], binormal_direction[1],
                      binormal_direction[2], length=1, normalize=True, color='gold', label="B")
            if verbose:
                print(f"Binormal vector: {binormal}")
                print(f"Binormal direction: {binormal_direction}")

        # Draw osculating circle
        if osculating_circle:
            oc = self.get_osculating_circle(p if p is not None else t_range[0])
            oc_points = self.parametrized_curve(oc, t_range, t_steps)
            # Plot with dashed lines
            ax.plot(oc_points[:, 0], oc_points[:, 1], oc_points[:, 2], linestyle='dashed', color='black',
                    label="Osculating circle")

        # Draw point on curve
        ax.scatter(point[0], point[1], point[2], color='black', label="p")

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

        ax.legend()

        plt.show()



class ReparametrizationMap():
    def __init__(self, map, var='t'):
        self.map = sp.sympify(map)
        self.var = var
        self.inv_map = self.get_inverse(self.map, self.var).subs('inv', self.var)

    # %%
    def get_inverse(self, expresssion, symbol):
        var = sp.Symbol(symbol)
        expr = sp.sympify(expresssion - sp.Symbol("inv"))
        return (sp.solve(expr, var)[-1])

    def reparametrize(self, curve):
        Y = curve.Y
        t_range = curve.t_range
        _t_steps = curve.t_steps
        _Y = [Y[0].subs('t', self.map), Y[1].subs('t', self.map), Y[2].subs('t', self.map)]
        _t_range = [self.inv_map.subs('t', t_range[0]), self.inv_map.subs('t', t_range[1])]
        _Y = Curve(_Y, _t_range, _t_steps)
        return _Y



if __name__ == "__main__":
    circular_helix = Curve(['cos(t)', 'sin(t)', 't'], t_range=[0, 2 * np.pi], t_steps=0.1)
    p = 2.1 * np.pi / 2

    # Visualize osculating circle
    """
    circular_helix.plot_curve(p=p, osculating_circle=True)
    """

    # Visualize repametrization
    """
    circular_helix.plot_curve(p=p, draw_tangent=True,
                               draw_normal=True, draw_binormal=True)
    repmap =  ReparametrizationMap('t+2', 't')
    circular_helix_ = repmap.reparametrize(circular_helix)
    circular_helix_.plot_curve(p=repmap.inv_map.subs(repmap.var, p), draw_tangent=True,
                              draw_normal=True, draw_binormal=True)
    """

