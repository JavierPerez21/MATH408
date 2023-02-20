from utils import *

class Surface():
    def __init__(self, surface, u_range=[0, 1], v_range=[0, 1], u_step=0.1, v_step=0.1):
        self.S = surface
        u = np.arange(u_range[0], u_range[1], u_step)
        v = np.arange(v_range[0], v_range[1], v_step)
        self.U = {'u': u, 'v': v}
        self.S_u = diff(self.S, 'u')
        self.S_v = diff(self.S, 'v')
        self.N_S = np.cross(self.S_u, self.S_v)

    def evaluate(self, x_expr, y_expr, z_expr, u, v):
        x = sp.sympify(x_expr)
        y = sp.sympify(y_expr)
        z = sp.sympify(z_expr)
        if isinstance(u, (int, float)) and isinstance(v, (int, float)):
            x_val = x.subs('u', u).subs('v', v)
            y_val = y.subs('u', u).subs('v', v)
            z_val = z.subs('u', u).subs('v', v)
        else:
            x_val = np.zeros((len(u), len(v)))
            y_val = np.zeros((len(u), len(v)))
            z_val = np.zeros((len(u), len(v)))
            for i in range(len(u)):
                for j in range(len(v)):
                    x_val[i, j] = x.subs('u', u[i]).subs('v', v[j])
                    y_val[i, j] = y.subs('u', u[i]).subs('v', v[j])
                    z_val[i, j] = z.subs('u', u[i]).subs('v', v[j])
        return np.array([x_val, y_val, z_val])

    def standard_unit_normal_vector(self, S=None):
        if S is None:
            S = self.S
        S_u = diff(S, 'u')
        S_v = diff(S, 'v')
        N_S = sp.simplify(np.cross(S_u, S_v))
        N_S = normalize(N_S)
        return N_S

    def plot_surface(self, S=None, u_range=None, v_range=None, u_step=None, v_step=None, p=None,
                     draw_normal=False, draw_tangents=False, verbose=False):
        if verbose:
            print("Plotting surface...")
        if S is None:
            S = self.S
        if u_range is None or v_range is None or u_step is None or v_step is None:
            U = self.U

        # Basics
        S_points = self.evaluate(S[0], S[1], S[2], U['u'], U['v'])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(S_points[0], S_points[1], S_points[2], cmap=plt.cm.YlGnBu_r)

        # Get point on curve
        if p is not None:
            point = self.evaluate(S[0], S[1], S[2], p[0], p[1])
        else:
            point = S_points[:, 0, 0]

        # Draw normal vector
        if draw_normal:
            normal = self.standard_unit_normal_vector(S)
            if p is not None:
                normal_direction = self.evaluate(normal[0], normal[1], normal[2], p[0], p[1])
            else:
                normal_direction = self.evaluate(normal[0], normal[1], normal[2], U['u'][0], U['v'][0])
            ax.quiver(point[0], point[1], point[2], normal_direction[0], normal_direction[1],
                      normal_direction[2], length=1, normalize=True, color='red')
            if verbose:
                print(f"Normal vector: {normal}")
                print(f"Normal direction: {normal_direction}")

        # Draw TpS
        if draw_tangents:
            S_u = diff(S, 'u')
            S_v = diff(S, 'v')
            if p is not None:
                TpS_u = self.evaluate(S_u[0], S_u[1], S_u[2], p[0], p[1])
                TpS_v = self.evaluate(S_v[0], S_v[1], S_v[2], p[0], p[1])
            else:
                TpS_u = self.evaluate(S_u[0], S_u[1], S_u[2], U['u'][0], U['v'][0])
                TpS_v = self.evaluate(S_v[0], S_v[1], S_v[2], U['u'][0], U['v'][0])
            ax.quiver(point[0], point[1], point[2], TpS_u[0], TpS_u[1], TpS_u[2],
                      length=1, normalize=True, color='orange')
            ax.quiver(point[0], point[1], point[2], TpS_v[0], TpS_v[1], TpS_v[2],
                      length=1, normalize=True, color='orange')
            if verbose:
                print(f"S_u: {S_u}")
                print(f"TpS_u: {TpS_u}")
                print(f"S_v: {S_v}")
                print(f"TpS_v: {TpS_v}")



        ax.scatter(point[0], point[1], point[2], color='black')

        # Write axes names
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()



if __name__ == "__main__":
    cylinder = Surface(['cos(u)', 'sin(u)', 'v'], u_range=[0, 2*np.pi], v_range=[-1, 1])
    cylinder.plot_surface(p=(np.pi/2, 0), draw_normal=True, draw_tangents=True, verbose=True)
