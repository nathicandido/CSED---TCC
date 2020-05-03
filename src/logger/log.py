from colorama import Fore, Style


class Log:

    def w(self, tag, message):
        print(f"{Fore.YELLOW + tag}: {message}")
        self.reset_color()

    def i(self, tag, message):
        print(f"{Fore.LIGHTBLACK_EX + tag}: {message}")
        self.reset_color()

    def d(self, tag, message):
        print(f"{Fore.BLUE + tag}: {message}")
        self.reset_color()

    def e(self, tag, message):
        print(f"{Fore.RED + tag}: {message}")
        self.reset_color()

    @staticmethod
    def reset_color():
        print(Style.RESET_ALL)
