import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

# class 정의
class Character:
    def __init__(self, name, health, damage, inventory):
        self.name = name
        self.health = health
        self.damage = damage
        self.inventory = inventory

    def __repr__(self):
        return self.name


# Character 클래스의 오브젝트 생성
heroes = []
heroes.append(Character('아이언맨', 100, 200, {'gold': 500, 'weapon': '레이저'}))
heroes.append(Character('데드풀', 300, 30, {'gold': 300, 'weapon': '장검'}))
heroes.append(Character('울버린', 200, 50, {'gold': 350, 'weapon': '클로'}))

monsters = []
monsters.append(Character('고블린', 90, 30, {'gold': 50, 'weapon': '창'}))
monsters.append(Character('드래곤', 200, 80, {'gold': 200, 'weapon': '화염'}))
monsters.append(Character('뱀파이어', 80, 120, {'gold': 1000, 'weapon': '최면술'}))

template = '{}'
print('# 히어로 리스트 확인')
print(heroes)

print('\n# 히어로 데이터 확인')
for hero in heroes:
    print(hero.__dict__)

print('\n# 몬스터 리스트 확인')
print(monsters)

print('\n# 몬스터 데이터 확인')
for monster in monsters:
    print(monster.__dict__)

del heroes[0]  # 히어로 리스트에서 아이언맨 삭제

print('\n# 히어로 리스트 재확인')
print(heroes)

print('# 히어로 데이터 재확인')
for hero in heroes:
    print(hero.__dict__)
