const expressionEl = document.getElementById('expression');
const currentEl = document.getElementById('current');

const OPERATOR_SYMBOLS = {
  add: '+',
  subtract: '−',
  multiply: '×',
  divide: '÷',
};

let currentValue = '0';
let previousValue = null;
let pendingOperator = null;
let awaitingOperand = false;
let justEvaluated = false;

function updateDisplay() {
  currentEl.textContent = formatForDisplay(currentValue);
  if (pendingOperator && previousValue !== null) {
    expressionEl.textContent = `${formatForDisplay(previousValue)} ${OPERATOR_SYMBOLS[pendingOperator]}`;
  } else {
    expressionEl.textContent = '';
  }
}

function formatForDisplay(value) {
  if (value === 'Error') return value;
  const num = Number(value);
  if (Number.isNaN(num)) return value;
  if (!Number.isFinite(num)) return 'Error';

  const isNegative = value.toString().startsWith('-');
  const [intPart, decPart] = value.toString().replace('-', '').split('.');
  const formattedInt = new Intl.NumberFormat('en-US').format(BigInt(intPart || '0'));
  const sign = isNegative ? '-' : '';
  return decPart !== undefined ? `${sign}${formattedInt}.${decPart}` : `${sign}${formattedInt}`;
}

function inputNumber(digit) {
  if (currentValue === 'Error' || awaitingOperand || justEvaluated) {
    currentValue = digit === '0' ? '0' : digit;
    awaitingOperand = false;
    if (justEvaluated) {
      previousValue = null;
      pendingOperator = null;
    }
    justEvaluated = false;
    updateDisplay();
    return;
  }
  if (currentValue === '0') {
    currentValue = digit;
  } else {
    if (currentValue.replace('-', '').replace('.', '').length >= 15) return;
    currentValue += digit;
  }
  updateDisplay();
}

function inputDecimal() {
  if (currentValue === 'Error' || awaitingOperand || justEvaluated) {
    currentValue = '0.';
    awaitingOperand = false;
    if (justEvaluated) {
      previousValue = null;
      pendingOperator = null;
    }
    justEvaluated = false;
    updateDisplay();
    return;
  }
  if (!currentValue.includes('.')) {
    currentValue += '.';
    updateDisplay();
  }
}

function inputBackspace() {
  if (currentValue === 'Error' || awaitingOperand || justEvaluated) {
    clearAll();
    return;
  }
  currentValue = currentValue.length > 1 ? currentValue.slice(0, -1) : '0';
  updateDisplay();
}

function clearAll() {
  currentValue = '0';
  previousValue = null;
  pendingOperator = null;
  awaitingOperand = false;
  justEvaluated = false;
  updateDisplay();
}

function negate() {
  if (currentValue === 'Error') return;
  if (currentValue !== '0') {
    currentValue = currentValue.startsWith('-') ? currentValue.slice(1) : `-${currentValue}`;
    updateDisplay();
  }
}

function percent() {
  if (currentValue === 'Error') return;
  currentValue = String(Number(currentValue) / 100);
  updateDisplay();
}

function compute(a, op, b) {
  switch (op) {
    case 'add':
      return a + b;
    case 'subtract':
      return a - b;
    case 'multiply':
      return a * b;
    case 'divide':
      return b === 0 ? NaN : a / b;
    default:
      return b;
  }
}

function setOperator(action) {
  if (currentValue === 'Error') return;

  if (pendingOperator !== null && !awaitingOperand) {
    const result = compute(Number(previousValue), pendingOperator, Number(currentValue));
    if (Number.isNaN(result) || !Number.isFinite(result)) {
      currentValue = 'Error';
      previousValue = null;
      pendingOperator = null;
      awaitingOperand = false;
      justEvaluated = true;
      updateDisplay();
      return;
    }
    previousValue = trimResult(result);
  } else {
    previousValue = currentValue;
  }

  pendingOperator = action;
  awaitingOperand = true;
  justEvaluated = false;
  currentValue = previousValue;
  updateDisplay();
}

function equals() {
  if (currentValue === 'Error' || pendingOperator === null || previousValue === null) return;

  const result = compute(Number(previousValue), pendingOperator, Number(currentValue));
  if (Number.isNaN(result) || !Number.isFinite(result)) {
    currentValue = 'Error';
  } else {
    currentValue = trimResult(result);
  }
  previousValue = null;
  pendingOperator = null;
  awaitingOperand = false;
  justEvaluated = true;
  updateDisplay();
}

function trimResult(num) {
  return parseFloat(num.toPrecision(12)).toString();
}

document.querySelector('.buttons').addEventListener('click', (event) => {
  const btn = event.target.closest('button');
  if (!btn) return;

  const { number, action } = btn.dataset;

  if (number !== undefined) {
    inputNumber(number);
    return;
  }

  switch (action) {
    case 'clear':
      clearAll();
      break;
    case 'negate':
      negate();
      break;
    case 'percent':
      percent();
      break;
    case 'decimal':
      inputDecimal();
      break;
    case 'backspace':
      inputBackspace();
      break;
    case 'equals':
      equals();
      break;
    case 'add':
    case 'subtract':
    case 'multiply':
    case 'divide':
      setOperator(action);
      break;
  }
});

const KEY_ACTIONS = {
  '+': 'add',
  '-': 'subtract',
  '*': 'multiply',
  '/': 'divide',
  Enter: 'equals',
  '=': 'equals',
  Backspace: 'backspace',
  Escape: 'clear',
  '%': 'percent',
  '.': 'decimal',
};

document.addEventListener('keydown', (event) => {
  if (event.key >= '0' && event.key <= '9') {
    inputNumber(event.key);
    return;
  }

  const action = KEY_ACTIONS[event.key];
  if (!action) return;

  event.preventDefault();
  switch (action) {
    case 'add':
    case 'subtract':
    case 'multiply':
    case 'divide':
      setOperator(action);
      break;
    case 'equals':
      equals();
      break;
    case 'backspace':
      inputBackspace();
      break;
    case 'clear':
      clearAll();
      break;
    case 'percent':
      percent();
      break;
    case 'decimal':
      inputDecimal();
      break;
  }
});

updateDisplay();
