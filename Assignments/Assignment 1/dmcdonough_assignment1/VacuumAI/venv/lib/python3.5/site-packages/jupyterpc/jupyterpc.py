from uncertainties import ufloat
from scipy import odr
import uncertainties as uc
from jinja2 import Template
import jinja2
import os

ucvar = [uc.core.AffineScalarFunc, uc.core.Variable]

def sci(num, decimals=4):
    if type(num) in ucvar:
        return num.format('L')
    elif type(num) in [float, int]:
        float_str = str("{0:."+str(decimals)+"e}").format(num)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            if int(exponent) in [-2, -1, 0, 1, 2]:
                return str(float(base)*10**int(exponent))
            return "{0} \\times 10^{{{1}}}".format(base, int(exponent))
        else:
            return float_str
    else:
        return str(num)

def ulist(list_val, sigma):
    if type(list_val[0]) not in [int, float]:
        raise TypeError('elements of list must be of type float or int, but are %s' % (type(list_val[0])))

    if type(sigma) in [int, float]:
        return [ufloat(x, sigma) for x in list_val]
    elif type(sigma) == list:
        if len(list_val) == len(sigma):
            return [ufloat(v, s) for v, s in zip(list_val, sigma)]
        else:
            raise BaseException('list and sigma must be the same length, but are %s and %s' % (len(list_val),len(sigma)))
    else:
        raise BaseException('sigma must be of type float, int or a list but is %s' % (type(sigma)))

def num(ulist):
    if type(ulist[0]) not in ucvar:
        raise BaseException('list has to be of type ulist')
    return [x.n for x in ulist]

def sig(ulist):
    if type(ulist[0]) not in ucvar:
        raise BaseException('list has to be of type ulist')
    return [x.s for x in ulist]

def fit(data_x, data_y, sigma_x=None, sigma_y=None, func=None, beta=[1., 0.], *args, **kwargs):
    if func == None:
        func = lambda p,x: p[0]*x+p[1]

    if type(data_x[0]) in ucvar:
        values_x = [d.n for d in data_x]
        sigma_x = [d.s if d.s!=0 else 1e-5 for d in data_y]
    elif type(data_x[0]) in [float, int]:
        values_x = data_x

    if type(data_y[0]) in ucvar:
        values_y = [d.n for d in data_y]
        sigma_y = [d.s if d.s!=0 else 1e-5 for d in data_y]
    elif type(data_y[0]) in [float, int]:
        values_y = data_y

    data = odr.RealData(values_x, values_y, sx=sigma_x, sy=sigma_y)
    model = odr.Model(func)
    odrfit = odr.ODR(data, model, beta0=beta)
    out = odrfit.run()
    return [ufloat(n, s) for n, s in zip(out.beta, out.sd_beta)] 

def table(name, data):
    max_len = max([len(x) for x in data.values()])
    tab_str = 'l'+''.join(['|l']*(len(data)-1))
    start = '''
            \\begin{table}
                \\caption{%s}
                \\centering
                \\begin{tabular}{%s}
            ''' % (str(name), tab_str)
    name_str = '\t\t'+''.join(['%s & ' % (n) for n in data.keys()])[:-2]+'\\\\\n'
    end = '''
                \\end{tabular}
            \\end{table}
          '''
    data_str = []
    for i in range(max_len):
        data_str += '\t\t\t'+''.join(['$'+sci(x[i])+'$ & ' for x in data.values()])[:-2]+'\\\\\n'
    data_str = ''.join(data_str)
    return start+name_str+data_str+end

def render(template_path, output_path, variables):
    latex_jinja_env = jinja2.Environment(
        block_start_string = '\BLOCK{',
        block_end_string = '}',
        variable_start_string = '\VAR{',
        variable_end_string = '}',
        comment_start_string = '\#{',
        comment_end_string = '}',
        line_statement_prefix = '%%',
        line_comment_prefix = '%#',
        trim_blocks = True,
        autoescape = False,
        loader = jinja2.FileSystemLoader(os.path.abspath('.'))
    )

    template = latex_jinja_env.get_template(template_path)
    with open(output_path, 'w') as out:
        out.write(template.render(variables))    

