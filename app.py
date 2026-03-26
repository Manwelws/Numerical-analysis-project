from methods.root_finding import Bisection, false_pos
from utils import parse_func

root_finding_functions = {"bisection": Bisection, "false-position": false_pos}


def main():

    user_method = input("choose a method bisection or false-position ")

    user_func = input("Enter func of x: ").strip()
    f, _ = parse_func(user_func)

    try:
        xl, xu = map(float, input("Enter xl and xu (space-separated): ").split())
    except ValueError:
        print("Invalid input , Enter xl and xu (space-separated):  ")
        return

    while True:
        flag = input("Stop by (i)terations or (e)psilon? ").strip().lower()

        if flag == "i":
            try:
                n = int(input("Enter number of iterations: "))
            except ValueError:
                print("Must be an integer.")
                continue
            method_class = root_finding_functions[user_method]
            method = method_class(f=f, xl=xl, xu=xu, tol=n)

        elif flag == "e":
            try:
                e = float(input("Enter epsilon (e.g. 0.0001): "))
            except ValueError:
                print("Must be a number.")
                continue
            method_class = root_finding_functions[user_method]
            method = method_class(f=f, xl=xl, xu=xu, tol=e)

        else:
            print("Invalid choice — enter 'i' or 'e'.")
            continue

        break

    try:
        result = method.solve()
    except ValueError as err:
        print(f"Err: {err}")
        return

    print(f"\nConverged: {result['converged']}")
    print(f"Root ≈ {result['root']:.8f}")
    print(f"Total iterations: {len(result['iterations'])}")

    print(
        f"\n{'Iter':<6} {'xl':<12} {'f_xl': <12} {'xu':<12} {'f_xu': <12} {'xr':<12} {'f_xr': <12} {'error':<12}"
    )
    print("-" * 54)
    for row in result["iterations"]:
        print(
            f"{row['iteration']:<6} {row['xl']:<12.6f} {row['f_xl']:<12.6f} {row['xu']:<12.6f} {row['f_xu']:<12.6f} {row['xr']:<12.6f} {row['f_xr']:<12.6f} {row['error']:<12.6f}"
        )


if __name__ == "__main__":
    main()
