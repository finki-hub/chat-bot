'use client';

/* Decorative sign-in showcase: a self-playing chat window that cycles through
   condensed real Q&A exchanges as if sent in one conversation. Entirely
   presentational — aria-hidden, pointer-events-none, nothing is interactive. */
import { useEffect, useRef, useState } from 'react';

type Exchange = {
  answer: string;
  question: string;
  source?: ExchangeSource;
};

type ExchangeSource = {
  title: string;
  type: 'документ' | 'прашање';
};

const EXCHANGES: Exchange[] = [
  {
    answer:
      'Треба да ги положиш сите задолжителни предмети и да освоиш 180 ЕКТС на тригодишни, односно 240 ЕКТС на четиригодишни студии, заедно со евидентирана студентска пракса. Потребно е и да го запишеш предметот „Дипломска работа“ и да одбраниш одобрена тема.',
    question: 'Кои се условите за дипломирање?',
    source: { title: 'Постапка за дипломирање', type: 'прашање' },
  },
  {
    answer:
      'Да, најдоцна две недели од почетокот на наставата. Барањето се поднесува во [[iKnow]] со коментар кој предмет со кој го заменуваш, а надоместот е 1.500 денари. За секој предмет се поднесува посебно барање.',
    question:
      'Дали можам да променам изборен предмет откако сум запишал семестар?',
    source: { title: 'Промена на предмет', type: 'прашање' },
  },
  {
    answer:
      'Нема услов за минимален број претходно освоени кредити. При уписот запишуваш најмалку 21 ЕКТС, најмногу 35 редовно, а до 40 со одобрена молба — само предмети за кои ги исполнуваш предусловите.',
    question: 'Колку кредити ми требаат за да запишам нареден семестар?',
    source: {
      title: 'Правилник за студии на прв и втор циклус · Член 16',
      type: 'документ',
    },
  },
  {
    answer:
      'Барањето се поднесува во [[iKnow]], во табот „Документи“, со уплата од 2.000 денари. Мирувањето е дозволено најмногу една академска година, а додека трае не можеш да запишуваш ниту да полагаш предмети.',
    question: 'Како да поднесам барање за мирување на студиите?',
    source: { title: 'Мирување на студии', type: 'прашање' },
  },
  {
    answer:
      'Непријавен испит можеш дополнително да го пријавиш за која било измината сесија, со „Барање за задоцнето пријавување на испит“ во [[iKnow]], во табот „Документи“. Казната е 1.000 денари за секој испит, а по уплатата ја известуваш Студентската служба.',
    question: 'Што се случува ако не пријавам испит во предвидениот рок?',
    source: {
      title: 'Закаснето пријавување испити (со казна)',
      type: 'прашање',
    },
  },
  {
    answer:
      'За семестар со 30 ЕКТС: 8.350 денари во државна и 14.500 во приватна квота за зимски семестар, односно 7.600 и 13.750 за летен. Точната партиципација за твојот упис е прикажана во [[iKnow]].',
    question: 'Колку изнесува трошокот при запишување на семестар?',
    source: {
      title: 'Уплата и трошоци при запишување на семестар',
      type: 'прашање',
    },
  },
  {
    answer:
      'Барањето се поднесува во [[iKnow]], во табот „Документи“. Електронското уверение е бесплатно, а хартиеното чини 100 денари и се подига според известувањето на студентската е-пошта.',
    question: 'Како да добијам уверение за редовен студент?',
    source: { title: 'Потврди и уверенија', type: 'прашање' },
  },
  {
    answer:
      'Студентската служба е во зградата на ТМФ, до кабинетот 117. Работи секој работен ден од 09:00 до 12:00 часот.',
    question: 'Каде се наоѓа студентската служба и кое е работното време?',
    source: { title: 'Студентска служба', type: 'прашање' },
  },
  {
    answer:
      'Праксата трае по еден месец за секоја студиска година освен првата — вкупно два месеца на тригодишни и три на четиригодишни студии. Се евидентира електронски преку [[порталот за студентски пракси]], каде компанијата ја внесува, а ти водиш дневник.',
    question: 'Како се евидентира студентската пракса и колку трае?',
    source: { title: 'Пракса', type: 'прашање' },
  },
  {
    answer:
      'Преминот не се смета за промена на студиска програма — изборот се прави пред одбраната на дипломската. Треба да ги положиш задолжителните предмети од тригодишната варијанта, да освоиш 180 ЕКТС и да имаш два месеца евидентирана пракса.',
    question:
      'Кои се условите за префрлање од четиригодишни на тригодишни студии?',
    source: {
      title: '3 (три) / 4 (четири) годишни студии',
      type: 'прашање',
    },
  },
  {
    answer:
      'Оперативни системи го држат десет професори, меѓу кои Боро Јакимовски, Невена Ацковска, Игор Мишковски, Ристе Стојанов и Сашо Граматиков.',
    question: 'Кои професори го држат предметот Оперативни системи?',
  },
  {
    answer:
      'Од базата на дипломски работи: детекција на упади со машинско учење, дијагноза на рана Алцхајмерова болест, предвидување туристички посети во Охрид и персонализирана препорака на фитнес-програми. Повеќе теми има на [[ФИНКИ Хаб — дипломски трудови]].',
    question:
      'Дај ми примери на дипломски теми од областа на машинското учење.',
  },
  {
    answer:
      'Предлог-наслов: „Развој и евалуација на прогресивна веб-апликација со микросервисна архитектура“. Предлог-комисија: Димитар Трајанов како ментор, со Сашо Граматиков и Ана Тодоровска како членови — врз основа на сродни дипломски трудови.',
    question:
      'Предложи ми наслов и комисија за дипломска од областа на веб-технологии.',
  },
  {
    answer:
      'Следна е есенската испитна сесија, во август и септември 2026. Распоредите се објавуваат во годишниот распоред на испити, а можеш да ги следиш и на [[Распореди на ФИНКИ Хаб]].',
    question: 'Кога е следната испитна сесија и каде се објавува распоредот?',
    source: {
      title: 'Правилник за студии на прв и втор циклус · Член 8',
      type: 'документ',
    },
  },
  {
    answer:
      'ФИНКИ Хаб е независна студентска иницијатива што прави алатки и ресурси за студентите на ФИНКИ — [[снимки од часови]], [[каталог на предмети]] и [[статистика за дипломски]]. Можеш да се вклучиш преку [[Discord-заедницата]] или со придонес на [[GitHub]].',
    question: 'Што е ФИНКИ Хаб и како да се вклучам во заедницата?',
    source: { title: 'Што е ФИНКИ Хаб', type: 'прашање' },
  },
];

type ChatMessage = {
  key: string;
  role: 'assistant' | 'user';
  source?: ExchangeSource;
  text: string;
};

const QUESTION_PAUSE_MS = 600;
const TYPING_MS = 1_300;
const ANSWER_DWELL_MS = 4_200;
const RESET_FADE_MS = 550;

const sleep = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const prefersReducedMotion = () =>
  typeof matchMedia === 'function' &&
  matchMedia('(prefers-reduced-motion: reduce)').matches;

const waitUntilVisible = async () => {
  if (!document.hidden) {
    return;
  }
  await new Promise<void>((resolve) => {
    const onChange = () => {
      if (document.hidden) {
        return;
      }

      document.removeEventListener('visibilitychange', onChange);
      resolve();
    };
    document.addEventListener('visibilitychange', onChange);
  });
};

/* Paces the loop: waits the given delay, then parks while the tab is hidden
   so exchanges never queue up in the background and dump on refocus. */
const pace = async (ms: number) => {
  await sleep(ms);
  await waitUntilVisible();
};

/* Renders [[text]] segments styled exactly like Streamdown's real message
   links (font-medium text-primary underline) but inert — plain spans inside
   a pointer-events-none window. */
const FAKE_LINK_PATTERN = /\[\[(?<label>[^[\]]+)\]\]/u;

const renderAnswer = (text: string) =>
  text.split(FAKE_LINK_PATTERN).map((part, index) =>
    index % 2 === 1 ? (
      <span
        className="font-medium text-primary underline"
        key={`${String(index)}-${part}`}
      >
        {part}
      </span>
    ) : (
      part
    ),
  );

const randomOrder = () => {
  const buffer = new Uint32Array(1);
  crypto.getRandomValues(buffer);
  return buffer[0] ?? 0;
};

const shuffled = <T,>(items: readonly T[]): T[] =>
  items
    .map((item) => ({ item, order: randomOrder() }))
    .sort((a, b) => a.order - b.order)
    .map(({ item }) => item);

const TypingDots = () => (
  <div className="flex items-center gap-1.5 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-500">
    <span className="size-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:-300ms]" />
    <span className="size-2 animate-bounce rounded-full bg-muted-foreground/60 [animation-delay:-150ms]" />
    <span className="size-2 animate-bounce rounded-full bg-muted-foreground/60" />
  </div>
);

export const ChatShowcase = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // slotReserved mounts the fixed-height indicator slot; typing fades the
  // dots in inside it without changing the layout
  const [slotReserved, setSlotReserved] = useState(false);
  const [typing, setTyping] = useState(false);
  const [fading, setFading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    const isAlive = () => alive;

    const playExchange = async (exchange: Exchange, index: number) => {
      // the indicator's space is reserved together with the question, so
      // the dots appearing later never shift the thread
      setMessages((current) => [
        ...current,
        { key: `${index}-q`, role: 'user', text: exchange.question },
      ]);
      setSlotReserved(true);
      await pace(QUESTION_PAUSE_MS);
      if (!isAlive()) {
        return;
      }
      setTyping(true);
      await pace(TYPING_MS);
      if (!isAlive()) {
        return;
      }
      setTyping(false);
      setSlotReserved(false);
      setMessages((current) => [
        ...current,
        {
          key: `${index}-a`,
          role: 'assistant',
          source: exchange.source,
          text: exchange.answer,
        },
      ]);
      await pace(ANSWER_DWELL_MS);
    };

    const resetThread = async () => {
      setFading(true);
      await pace(RESET_FADE_MS);
      if (!isAlive()) {
        return;
      }
      setMessages([]);
      setFading(false);
      await pace(400);
    };

    const run = async () => {
      // brief beat so the window is visible before the first message lands
      await pace(700);

      while (isAlive()) {
        for (const [index, exchange] of shuffled(EXCHANGES).entries()) {
          if (!isAlive()) {
            return;
          }
          await playExchange(exchange, index);
        }

        if (!isAlive()) {
          return;
        }
        await resetThread();
      }
    };

    const [first] = EXCHANGES;

    if (prefersReducedMotion() && first !== undefined) {
      setMessages([
        { key: 'static-q', role: 'user', text: first.question },
        {
          key: 'static-a',
          role: 'assistant',
          source: first.source,
          text: first.answer,
        },
      ]);
    } else {
      void run();
    }

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (el === null || typeof el.scrollTo !== 'function') {
      return;
    }
    el.scrollTo({
      behavior: prefersReducedMotion() ? 'auto' : 'smooth',
      top: el.scrollHeight,
    });
    // deliberately not keyed on the typing state: the indicator fades into
    // space reserved at send time, so it must never trigger a scroll
  }, [messages]);

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none flex h-full select-none flex-col overflow-hidden rounded-2xl border border-border bg-card"
    >
      <div className="flex shrink-0 items-center gap-2.5 border-b border-border/60 px-4 py-3">
        <img
          alt=""
          className="h-6 w-6 object-contain"
          height={24}
          src="/logo.png"
          width={24}
        />
        <span className="text-sm font-semibold tracking-tight">
          ФИНКИ Хаб / Чат
        </span>
      </div>

      <div
        className="min-h-0 flex-1 overflow-y-auto px-4 pb-6 pt-5 [scrollbar-width:none] sm:px-5 [&::-webkit-scrollbar]:hidden"
        ref={scrollRef}
      >
        <div
          className={`space-y-4 transition-opacity duration-500 ${fading ? 'opacity-0' : 'opacity-100'}`}
        >
          {messages.map((message) =>
            message.role === 'user' ? (
              <div
                className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-4 py-2.5 text-sm leading-relaxed shadow-sm motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500"
                key={message.key}
              >
                {message.text}
              </div>
            ) : (
              <div
                className="max-w-full space-y-3 text-sm leading-relaxed motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500"
                key={message.key}
              >
                <p>{renderAnswer(message.text)}</p>
                {message.source === undefined ? null : (
                  <div className="rounded-lg border border-border/70 bg-muted/20 p-3 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:fill-mode-both stagger-1">
                    <span className="rounded-full border border-border/70 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      {message.source.type}
                    </span>
                    <p className="mt-2 line-clamp-2 text-sm font-medium leading-snug text-foreground">
                      {message.source.title}
                    </p>
                  </div>
                )}
              </div>
            ),
          )}
          {slotReserved ? (
            <div className="flex h-8 items-center">
              {typing ? <TypingDots /> : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
