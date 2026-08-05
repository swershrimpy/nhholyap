import torch
from torch import nn
from utils import *


torch.set_default_dtype(torch.float32)
# torch.manual_seed(0)

def FFNetwork(input, output, intermediate_layers, sigmoid=False, constant_f=False, constant_B=False, activation=None):
    """
    :param input: input dimension of network (nx x nu)
    :param output: output dimension of network (nx)
    :param intermediate_layers: number of hidden layers
    :param sigmoid: if you want to put a sigmoid on the output
    :param constant_f: if f(x) is a constant vector in R^{nx}
    :param constant_B: if B(x) is a constant matrix in R^{nx x nu}
    :param activation: if you have custom activation functions; else, default to ReLU
    :return: custom network
    """
    linears = []
    if constant_f:
        return torch.nn.Parameter(torch.rand(output, output).float())
    if constant_B:
        dim_z = input
        dim_u = int(output / input)
        if dim_z == dim_u:
            return torch.nn.Parameter(torch.eye(int(torch.sqrt(output))).float())
        else:
            return torch.nn.Parameter(torch.cat([torch.zeros(dim_z-dim_u, dim_u), 0.75*torch.eye(dim_u)], dim=0).float())
    for i in range(len(intermediate_layers) + 1):
        if i == 0:
            linears.append(torch.nn.Linear(input, intermediate_layers[i]).float())
            if activation is not None:
                linears.append(activation)
            else:
                linears.append(torch.nn.ReLU())
        elif i == len(intermediate_layers):
            linears.append(torch.nn.Linear(intermediate_layers[i - 1], output).float())
        else:
            linears.append(torch.nn.Linear(intermediate_layers[i - 1], intermediate_layers[i]).float())
            if activation is not None:
                linears.append(activation)
            else:
                linears.append(torch.nn.ReLU())
    if sigmoid:
        linears.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*linears)

class PolynomialNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, degree, lim=1000., monomials=None, sc=False):
        super(PolynomialNetwork, self).__init__()
        z_str, z_out = gen_z_mon(input_dim, degree)
        self.z_str = z_str
        self.lim = lim
        self.sc = sc
        # Instantiate coefficients of monomials
        if z_str == '[]':
            self.coeffs = nn.Parameter(torch.randn(output_dim, 1))
        else:
            self.coeffs = nn.Parameter(torch.randn(output_dim, len(z_out)))
    def forward(self, x):
        if len(eval(self.z_str)) == 0:
            if self.sc:
                return (torch.vstack([torch.zeros(2,1).to(self.coeffs.device),
                                      self.coeffs[2:].clamp(min=-10, max=10)]) @ torch.ones((1,
                            x.shape[0])).to(x.device)).T.clamp(min=-self.lim, max=self.lim)
            else:
                return (self.coeffs.clamp(min=-10,max=10) @ torch.ones((1,
                                  x.shape[0])).to(x.device)).T.clamp(min=-self.lim, max=self.lim)
        else:
            return (self.coeffs.clamp(min=-10,max=10) @ z_mon_eval(x, self.z_str)).T.clamp(min=-self.lim, max=self.lim)

class PolynomialNetwork_t(nn.Module):
    def __init__(self, input_dim, output_dim, degree, lim=1000., monomials=None, sc=False):
        super(PolynomialNetwork_t, self).__init__()
        z_str, z_out = gen_z_mon(input_dim, degree)
        z_str = '[x[:,0], x[:,1], x[:,2], x[:,0]*x[:,1], x[:,0]*x[:,2], x[:,1]*x[:,2]]'
        self.z_str = z_str
        self.lim = lim
        self.sc = sc
        # Instantiate coefficients of monomials
        if z_str == '[]':
            self.coeffs = nn.Parameter(torch.randn(output_dim, 1))
        else:
            # self.coeffs = nn.Parameter(torch.randn(output_dim, len(z_out)))
            self.coeffs = nn.Parameter(torch.randn(output_dim, 6))
    def forward(self, t, x):
        if len(eval(self.z_str)) == 0:
            if self.sc:
                return (torch.vstack([torch.zeros(2,1).to(self.coeffs.device),
                                      self.coeffs[2:].clamp(min=-10, max=10)]) @ torch.ones((1,
                            x.shape[0])).to(x.device)).T.clamp(min=-self.lim, max=self.lim)
            else:
                return (self.coeffs.clamp(min=-10,max=10) @ torch.ones((1,
                          x.shape[0])).to(x.device)).T.clamp(min=-self.lim, max=self.lim)
        else:
            if self.sc:
                return (torch.vstack([torch.tensor([[0., 0., 0., 0., 0., 1.],
                                                    [0., 0., 0., 0., -1., 0.]]).to(self.coeffs.device),
                                      self.coeffs[2:].clamp(min=-10, max=10)]) @ z_mon_eval(x.reshape(-1,x.shape[-1]),
                                             self.z_str)).T.clamp(min=-self.lim, max=self.lim).reshape(x.shape)
            else:
                return (self.coeffs.clamp(min=-10,max=10) @ z_mon_eval(x.reshape(-1,x.shape[-1]),
                                             self.z_str)).T.clamp(min=-self.lim, max=self.lim).reshape(x.shape)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Reconstruct(nn.Module):
    """
    Encoder, decoder, no dynamics
    """
    def __init__(self, z_dim):
        super(Reconstruct, self).__init__()
        x_channels = 1

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, z_dim), nn.Tanh()
        )
        self.decoder = nn.Sequential(nn.Linear(z_dim, 512),
            nn.ReLU(), nn.Linear(512, 128), nn.ReLU(),
            View((-1, 8, 4, 4)),
            nn.ConvTranspose2d(8, 4, 7, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 6, stride=2),
            nn.Sigmoid()
        )

    def encode(self, y):
        return self.encoder(y)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, y):
        z = self.encoder(y)
        return self.decoder(z), z

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class ContrastiveModel(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(ContrastiveModel, self).__init__()
        x_channels = 1

        # 64x64 architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=x_channels, out_channels=4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=6, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, params.y_red_dim), nn.Tanh()
        )
        self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
             nn.ReLU(), nn.Linear(512, 128), nn.ReLU(),
             View((-1, 8, 4, 4)),
             nn.ConvTranspose2d(8, 4, 7, stride=2),
             nn.ReLU(),
             nn.ConvTranspose2d(4, 4, 6, stride=2),
             nn.ReLU(),
             nn.ConvTranspose2d(4, 1, 6, stride=2),
             nn.Sigmoid()
        )
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, params.y_red_dim)#, nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
        #      nn.ReLU(), nn.Linear(512, 4096),
        #      nn.Sigmoid()
        # )

        # 16x16 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, params.y_red_dim)#, nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
        #      nn.ReLU(), nn.Linear(512, 256),
        #      nn.Sigmoid()
        # )

        self.f = PolynomialNetwork(params.z_dim, params.z_dim, params.f_degree, lim=params.lim)
        self.g = PolynomialNetwork(params.z_dim, params.z_dim * params.u_dim, params.g_degree, lim=params.lim)
        self.h = PolynomialNetwork(params.z_dim, params.y_red_dim, params.h_degree, lim=params.lim)
        self.dt = params.dt
        # self.f = FFNetwork(params_f.input, params_f.output, params_f.intermediate_layers, sigmoid=params_f.sigmoid,
        #                    constant_f=params_f.constant_f, constant_B=params_f.constant_B, activation=params_f.activation)
        # self.g = FFNetwork(params_g.input, params_g.output, params_g.intermediate_layers, sigmoid=params_g.sigmoid,
        #                    constant_f=params_g.constant_f, constant_B=params_g.constant_B, activation=params_g.activation)

    def encode(self, y):
        return self.encoder(y)

    def decode(self, y_red):
        return self.decoder(y_red)

    def forward_z(self, z, u):
        z_dot = self.f(z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)
        return z_tp1, y_red_tp1

    def forward_z_horizon(self, z_init, u):
        """
        :param z_init: in shape (batch_size, z_dim)
        :param u: in shape (batch_size, horizon, u_dim)
        :return: z_propagated: in shape (batch_size, horizon, z_dim)
        """
        zs = [z_init]
        ys = [self.h(zs[-1])]
        for t in range(u.shape[1]-1):
            z_tp1, y_tp1 = self.forward_z(zs[-1], u[:, t, :])
            zs.append(z_tp1)
            ys.append(y_tp1)
        zs = torch.stack(zs, dim=1)
        ys = torch.stack(ys, dim=1)
        return zs, ys

    def forward(self, y, u):
        z = self.encoder(y)
        z_dot = self.f(z) + self.g(z) @ u
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)
        return z_tp1, y_red_tp1

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedError(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedError, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            # nn.Linear(params.z_dim, params.hidden_dim), nn.Tanh(), nn.Linear(params.hidden_dim, params.hidden_dim), nn.Tanh(),
            nn.Linear(params.z_dim, params.hidden_dim), nn.Tanh(), nn.Linear(params.hidden_dim, params.hidden_dim),
            nn.Tanh(), nn.Linear(params.hidden_dim, 1) , nn.Sigmoid()
        )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoError(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoError, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.x_dim_error, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 1)#, nn.Tanh()
        )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoErrorObservability(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoErrorObservability, self).__init__()
        x_channels = 1
        # self.net = nn.Sequential(
        #     nn.Linear(params.x_dim_error, 256), nn.GELU(approximate='tanh'),
        #     nn.Linear(256, 256), nn.GELU(approximate='tanh'),
        #     nn.Linear(256, 1)  # , nn.Softplus()
        # )
        self.net = nn.Sequential(
            nn.Linear(params.x_dim_error, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 1)#, nn.Softplus()
        )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDino(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDino, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, params.y_red_dim)#, nn.Tanh()
        )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservability(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoObservability, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, params.y_red_dim)  # , nn.Tanh()
        )
        self.C = nn.Parameter(torch.randn(params.y_red_dim, params.x_dim))
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservabilityLarge(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoObservabilityLarge, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
            # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(1024, params.y_red_dim)  # , nn.Tanh()
        )
        self.C = nn.Parameter(torch.randn(params.y_red_dim, params.x_dim))
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservabilityLargeUnicycle(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoObservabilityLargeUnicycle, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, params.y_red_dim)  # , nn.Tanh()
        )
        self.C = nn.Parameter(torch.randn(params.y_red_dim, params.x_dim))
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservabilityBoundQuadrotor(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params=None, multi=False):
        super(SupervisedDinoObservabilityBoundQuadrotor, self).__init__()
        x_channels = 1
        if params is None:
            if multi:
                last_dim = 5
            else:
                last_dim = 1
            self.net = nn.Sequential(
                nn.Linear(5, 512), nn.Mish(),  # nn.Tanh(),  # nn.GELU(approximate='tanh'),
                nn.Linear(512, 512), nn.Mish(),  # nn.GELU(approximate='tanh'),
                nn.Linear(512, 512), nn.Mish(),  # nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),  # nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),  # nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
                nn.Linear(512, last_dim)  # , nn.Tanh()
            )
            # self.net = nn.Sequential(
            #     nn.Linear(5, 1024), nn.Mish(),#nn.Tanh(),  # nn.GELU(approximate='tanh'),
            #     nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
            #     # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            #     nn.Linear(1024, last_dim)  # , nn.Tanh()
            # )
        else:
            self.net = nn.Sequential(
                nn.Linear(params.y_red_dim, 1024), nn.Tanh(),#nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1)  # , nn.Tanh()
            )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservabilityBoundRope(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params=None, multi=False):
        super(SupervisedDinoObservabilityBoundRope, self).__init__()
        x_channels = 1
        if params is None:
            if multi:
                last_dim = 21
            else:
                last_dim = 1
            self.net = nn.Sequential(
                nn.Linear(21, 1024), nn.Mish(),#nn.Tanh(),  # nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(1024, 1024), nn.Mish(),#nn.GELU(approximate='tanh'),
                # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
                nn.Linear(1024, last_dim)  # , nn.Tanh()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(params.y_red_dim, 1024), nn.Tanh(),#nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1024), nn.GELU(approximate='tanh'),
                # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
                nn.Linear(1024, 1)  # , nn.Tanh()
            )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class SupervisedDinoObservabilityLargeQuadrotor(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoObservabilityLargeQuadrotor, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            # nn.Linear(512, 512), nn.GELU(approximate='tanh'),
            nn.Linear(512, params.y_red_dim)  # , nn.Tanh()
        )
        self.C = nn.Parameter(torch.randn(params.y_red_dim, params.x_dim))
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))
class SupervisedDinoLinear(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(SupervisedDinoLinear, self).__init__()
        x_channels = 1
        self.net = nn.Sequential(
            nn.Linear(params.z_dim, params.y_red_dim)#, nn.Tanh()
        )
    def forward(self, y):
        return self.net(y)

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))
class ContrastiveModelKnown(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(ContrastiveModelKnown, self).__init__()
        x_channels = 1
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, params.y_red_dim)#, nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
        #      nn.ReLU(), nn.Linear(512, 4096),
        #      nn.Sigmoid()
        # )

        # # 16x16 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(256, 512), nn.LeakyReLU(), nn.Linear(512, 512), nn.LeakyReLU(),
        #     nn.Linear(512, params.y_red_dim)#, nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
        #      nn.Linear(512, 512), nn.LeakyReLU(),
        #      nn.LeakyReLU(), nn.Linear(512, 256),
        #      nn.Sigmoid()
        # )
        # 24x24 architecture
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(576, 1024), nn.ELU(), nn.Linear(1024, 512), nn.ELU(),
            nn.Linear(512, params.y_red_dim)  # , nn.Tanh()
        )
        self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 512),
             nn.Linear(512, 1024), nn.ELU(),
             nn.ELU(), nn.Linear(1024, 576),
             nn.Sigmoid()
        )

        self.f = PolynomialNetwork(params.z_dim, params.z_dim, params.f_degree, lim=params.lim)
        self.g = PolynomialNetwork(params.z_dim, params.z_dim * params.u_dim, params.g_degree, lim=params.lim)
        self.h = PolynomialNetwork(params.z_dim, params.y_red_dim, params.h_degree, lim=params.lim)
        self.dt = params.dt

    def encode(self, y):
        return self.encoder(y)

    def decode(self, y_red):
        return self.decoder(y_red)

    def forward_z(self, z, u):
        z_dot = self.f(z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)/self.h.coeffs.norm()
        return z_tp1, y_red_tp1

    def forward_z_horizon(self, z_init, u):
        """
        :param z_init: in shape (batch_size, z_dim)
        :param u: in shape (batch_size, horizon, u_dim)
        :return: z_propagated: in shape (batch_size, horizon, z_dim)
        """
        zs = [z_init]
        ys = [self.h(zs[-1])/self.h.coeffs.norm()]
        for t in range(u.shape[1]-1):
            z_tp1, y_tp1 = self.forward_z(zs[-1], u[:, t, :])
            zs.append(z_tp1)
            ys.append(y_tp1)
        zs = torch.stack(zs, dim=1)
        ys = torch.stack(ys, dim=1)
        return zs, ys

    def forward(self, y, u):
        z = self.encoder(y)
        z_dot = self.f(z) + self.g(z) @ u
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)
        return z_tp1, y_red_tp1

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class ContrastiveModelKnownCT(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(ContrastiveModelKnownCT, self).__init__()
        x_channels = 1

        # # # 16x16 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2*256, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #      nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #      nn.Linear(1024, 256),
        #      nn.Sigmoid()
        # )
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2),
        #     # nn.LayerNorm((4, 10, 10)),
        #     nn.Mish(),
        #     Flatten(),
        #     nn.Linear(144, 128), nn.Mish(), nn.Linear(128, params.z_dim)
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #      nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #      nn.Linear(1024, 256),
        #      View((-1, 1, 16, 16)),
        #      nn.Sigmoid()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 9*16),
        #     nn.Mish(),
        #     View((-1, 16, 3, 3)),
        #     nn.ConvTranspose2d(16, 16, (5, 5), stride=(3, 3)),
        #     nn.LayerNorm((16, 11, 11)),
        #     nn.Mish(),
        #     nn.ConvTranspose2d(16, 12, (3, 3), stride=(2, 2)),
        #     nn.LayerNorm((12, 23, 23)),
        #     nn.Mish(),
        #     nn.ConvTranspose2d(12, 1, (2, 2), stride=(1, 1)),
        #     Flatten(),
        #     nn.Sigmoid()
        # )

        # # 24x24 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2*576, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #      nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #      nn.Linear(1024, 576),
        #      nn.Sigmoid()
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2),
            # nn.LayerNorm((4, 10, 10)),
            nn.Mish(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=6, stride=2),
            # nn.LayerNorm((4, 3, 3)),
            nn.Mish(),
            Flatten(),
            nn.Linear(36, 128), nn.Mish(), nn.Linear(128, params.enc_dim)
        )
        self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 9*16),
            nn.Mish(),
            View((-1, 16, 3, 3)),
            nn.ConvTranspose2d(16, 16, (5, 5), stride=(3, 3)),
            nn.LayerNorm((16, 11, 11)),
            nn.Mish(),
            nn.ConvTranspose2d(16, 12, (3, 3), stride=(2, 2)),
            nn.LayerNorm((12, 23, 23)),
            nn.Mish(),
            nn.ConvTranspose2d(12, 1, (2, 2), stride=(1, 1)),
            Flatten(),
            nn.Sigmoid()
        )

        # # 32x32 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2 * 1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #                              nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #                              nn.Linear(1024, 1024),
        #                              nn.Sigmoid()
        #                              )

        # 48x48 architecture
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5, stride=2),
        #     # nn.LayerNorm((4, 10, 10)),
        #     nn.Mish(),
        #     nn.Conv2d(in_channels=4, out_channels=4, kernel_size=5, stride=2),
        #     # nn.LayerNorm((4, 3, 3)),
        #     nn.Mish(),
        #     Flatten(),
        #     nn.Linear(324, 128), nn.Mish(), nn.Linear(128, params.z_dim)
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 9*16),
        #     nn.Mish(),
        #     nn.Linear(144, 1024), nn.Mish(),
        #     nn.Linear(1024, 2304),
        #     nn.Sigmoid()
        # )

        self.f = PolynomialNetwork_t(params.z_dim, params.z_dim, params.f_degree, lim=params.lim, sc=params.sc)
        self.g = PolynomialNetwork(params.z_dim, params.z_dim * params.u_dim, params.g_degree, lim=params.lim, sc=params.sc)
        self.h = PolynomialNetwork(params.z_dim, params.y_red_dim, params.h_degree, lim=params.lim)
        self.dt = params.dt

    def encode(self, y):
        return self.encoder(y)

    def decode(self, y_red):
        return self.decoder(y_red)

    def dynamics_deriv_eval(self, t, z):
        u = torch.sin(2*np.pi*t)[:,None]
        z_dot = self.f(t, z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        return z_dot

    def forward_z(self, z, u):
        z_dot = self.f(z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)/self.h.coeffs.norm()
        return z_tp1, y_red_tp1

    def forward_z_horizon(self, z_init, u):
        """
        :param z_init: in shape (batch_size, z_dim)
        :param u: in shape (batch_size, horizon, u_dim)
        :return: z_propagated: in shape (batch_size, horizon, z_dim)
        """
        zs = [z_init]
        ys = [self.h(zs[-1])/self.h.coeffs.norm()]
        for t in range(u.shape[1]-1):
            z_tp1, y_tp1 = self.forward_z(zs[-1], u[:, t, :])
            zs.append(z_tp1)
            ys.append(y_tp1)
        zs = torch.stack(zs, dim=1)
        ys = torch.stack(ys, dim=1)
        return zs, ys

    def forward(self, y, u):
        z = self.encoder(y)
        z_dot = self.f(z) + self.g(z) @ u
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)
        return z_tp1, y_red_tp1

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))

class ContrastiveModelNN(nn.Module):
    """
    Encoder, dynamics
    """
    def __init__(self, params):
        super(ContrastiveModelNN, self).__init__()
        x_channels = 1

        # # # 16x16 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2*256, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #      nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #      nn.Linear(1024, 256),
        #      nn.Sigmoid()
        # )

        # # 24x24 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2*576, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #      nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #      nn.Linear(1024, 576),
        #      nn.Sigmoid()
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2),
            # nn.LayerNorm((4, 10, 10)),
            nn.Mish(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=6, stride=2),
            # nn.LayerNorm((4, 3, 3)),
            nn.Mish(),
            Flatten(),
            nn.Linear(36, 128), nn.Mish(), nn.Linear(128, params.z_dim)
        )
        self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 9*16),
            nn.Mish(),
            View((-1, 16, 3, 3)),
            nn.ConvTranspose2d(16, 16, (5, 5), stride=(3, 3)),
            nn.LayerNorm((16, 11, 11)),
            nn.Mish(),
            nn.ConvTranspose2d(16, 12, (3, 3), stride=(2, 2)),
            nn.LayerNorm((12, 23, 23)),
            nn.Mish(),
            nn.ConvTranspose2d(12, 1, (2, 2), stride=(1, 1)),
            Flatten(),
            nn.Sigmoid()
        )

        # # 32x32 architecture
        # self.encoder = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(2 * 1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, params.z_dim)  # , nn.Tanh()
        # )
        # self.decoder = nn.Sequential(nn.Linear(params.y_red_dim, 1024),
        #                              nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
        #                              nn.Linear(1024, 1024),
        #                              nn.Sigmoid()
        #                              )

        self.f = nn.Sequential(nn.Linear(params.z_dim, 128),
                                     nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, params.z_dim)
                                     )
        self.g = nn.Sequential(nn.Linear(params.z_dim, 128),
                                     nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, params.z_dim)
                                     )
        self.h = nn.Sequential(nn.Linear(params.z_dim, 128),
                                     nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, params.y_red_dim)
                                     )
        self.dt = params.dt

    def encode(self, y):
        return self.encoder(y)

    def decode(self, y_red):
        return self.decoder(y_red)

    def dynamics_deriv_eval(self, t, z):
        u = torch.sin(2*np.pi*t)[:,None]
        z_dot = self.f(z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        return z_dot

    def forward_z(self, z, u):
        z_dot = self.f(z) + (self.g(z).view(z.shape[0], z.shape[1], u.shape[1]) @ u[:,:,None]).squeeze(-1)
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)/self.h.coeffs.norm()
        return z_tp1, y_red_tp1

    def forward_z_horizon(self, z_init, u):
        """
        :param z_init: in shape (batch_size, z_dim)
        :param u: in shape (batch_size, horizon, u_dim)
        :return: z_propagated: in shape (batch_size, horizon, z_dim)
        """
        zs = [z_init]
        ys = [self.h(zs[-1])/self.h.coeffs.norm()]
        for t in range(u.shape[1]-1):
            z_tp1, y_tp1 = self.forward_z(zs[-1], u[:, t, :])
            zs.append(z_tp1)
            ys.append(y_tp1)
        zs = torch.stack(zs, dim=1)
        ys = torch.stack(ys, dim=1)
        return zs, ys

    def forward(self, y, u):
        z = self.encoder(y)
        z_dot = self.f(z) + self.g(z) @ u
        z_tp1 = z + self.dt * z_dot
        y_red_tp1 = self.h(z_tp1)
        return z_tp1, y_red_tp1

    def save_network_parameters(self, filename):
        torch.save(self.state_dict(), filename)

    def load_network_parameters(self, filename):
        self.load_state_dict(torch.load(filename))